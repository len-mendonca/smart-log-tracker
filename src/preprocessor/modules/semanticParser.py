import re
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class SemanticOpenLogParser:
    def __init__(self,
                 llm_model="gpt-4o-mini",
                 embedding_model="text-embedding-3-small",
                 sim_threshold=0.9,
                 max_wildcard_density=0.3):
        """
        :param sim_threshold: cosine threshold for merging semantic clusters
        :param max_wildcard_density: max proportion of <*> tokens in template
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0, top_p=0.1)
        self.embedder = OpenAIEmbeddings(model=embedding_model)
        self.sim_threshold = sim_threshold
        self.max_wildcard_density = max_wildcard_density
        self.template_registry = {}  # in-memory (replace with DB later)

    # -----------------------
    # STEP 1: Structural Grouping
    # -----------------------
    def structural_signature(self, masked_log: str):
        tokens = masked_log.split()
        sig = " ".join("<*>" if t.startswith("<") and t.endswith(">") else t for t in tokens)
        return sig

    def group_by_structure(self, masked_logs):
        groups = defaultdict(list)
        for log in masked_logs:
            sig = self.structural_signature(log)
            groups[sig].append(log)
        return groups

    # -----------------------
    # STEP 2: Semantic Merge
    # -----------------------
    def embed_text(self, text):
        return np.array(self.embedder.embed_query(text))
    
    def token_overlap(a, b):
        toks_a = set(a.replace("<*>", "").split())
        toks_b = set(b.replace("<*>", "").split())
        if not toks_a or not toks_b:
           return 0
        return len(toks_a & toks_b) / max(len(toks_a), len(toks_b))

    def merge_semantic_clusters(self, groups, alpha=0.7):
        signatures = list(groups.keys())
        vecs = np.array([self.embed_text(sig) for sig in signatures])
        sims = cosine_similarity(vecs)
        merged, used = [], set()

        for i, sig in enumerate(signatures):
            if sig in used:
               continue
            cluster = [sig]
            used.add(sig)
            for j in range(i + 1, len(signatures)):
               sem_sim = sims[i, j]
               tok_sim = self.token_overlap(sig, signatures[j])
               final_sim = alpha * sem_sim + (1 - alpha) * tok_sim

               if final_sim > self.sim_threshold and tok_sim > 0.4:
                   cluster.append(signatures[j])
                   used.add(signatures[j])
            merged.append(cluster)
        return merged

    # -----------------------
    # STEP 3: LLM Template Refinement
    # -----------------------
    def generate_template(self, logs):
        template_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a deterministic log template generator. Use <*> for dynamic fields."),
            ("human", f"Generate a single canonical template for these logs:\n" +
                      "\n".join(f"{i+1}. {log}" for i, log in enumerate(logs)))
        ])
        response = self.llm.invoke(template_prompt)
        template = response.content.strip()

        # Basic guardrails
        if template.count("<*>") / max(len(template.split()), 1) > self.max_wildcard_density:
            raise ValueError("Too many wildcards in template")

        return template

    def validate_template(self, template, logs):
        """Check regex match rate >= 95%"""
        regex = re.escape(template).replace(r'\<\*\>', r'.*?')
        pattern = re.compile(f"^{regex}$")
        matched = sum(1 for l in logs if pattern.match(l))
        return matched / len(logs) >= 0.95

    # -----------------------
    # STEP 4: Anchor Extraction
    # -----------------------
    def extract_anchor_words(self, logs):
        """Extract meaningful verbs/adjectives to capture semantic intent."""
        anchors = set()
        for log in logs:
            words = [w for w in re.findall(r'\b[a-zA-Z]+\b', log) if len(w) > 3]
            anchors.update(words)
        return list(anchors)

    def centroid_embedding(self, anchors):
        """Compute mean embedding of anchor words."""
        if not anchors:
            return np.zeros(768)
        vecs = np.array([self.embed_text(a) for a in anchors])
        return np.mean(vecs, axis=0)

    # -----------------------
    # Pipeline
    # -----------------------
    def fit(self, masked_logs):
        """Generate templates and semantic anchors from good logs."""
        structural_groups = self.group_by_structure(masked_logs)
        semantic_clusters = self.merge_semantic_clusters(structural_groups)

        for cluster in semantic_clusters:
            # Merge logs from all similar groups
            merged_logs = []
            for sig in cluster:
                merged_logs.extend(structural_groups[sig])

            template = self.generate_template(merged_logs)
            if not self.validate_template(template, merged_logs):
                continue

            anchors = self.extract_anchor_words(merged_logs)
            centroid = self.centroid_embedding(anchors)

            template_id = f"T{len(self.template_registry)+1:03d}"
            self.template_registry[template_id] = {
                "template": template,
                "anchors": anchors,
                "centroid": centroid,
                "count": len(merged_logs)
            }

        return self.template_registry

    # -----------------------
    # STEP 5: Runtime Anomaly Detection
    # -----------------------
    def detect_semantic_anomaly(self, log):
        """Check if new log’s dynamic anchor is semantically aligned."""
        matched_template = None
        for tid, entry in self.template_registry.items():
            regex = re.escape(entry["template"]).replace(r'\<\*\>', r'.*?')
            if re.match(f"^{regex}$", log):
                matched_template = tid
                break

        if not matched_template:
            return {"status": "unmatched", "reason": "No template found"}

        # Extract dynamic word(s)
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', log) if len(w) > 3]
        anchor_vecs = np.array([self.embed_text(w) for w in words])
        anchor_centroid = np.mean(anchor_vecs, axis=0)

        sim = cosine_similarity(
            [anchor_centroid],
            [self.template_registry[matched_template]["centroid"]]
        )[0][0]

        if sim < 0.75:
            return {"status": "semantic_anomaly", "similarity": sim, "template_id": matched_template}
        return {"status": "normal", "similarity": sim, "template_id": matched_template}


