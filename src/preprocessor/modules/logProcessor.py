import re
import ipaddress

class LogPreprocessor:
    """
    Masks dynamic data (timestamps, UUIDs, IPs, durations, etc.)
    and stores extracted values by positional order.
    """

    def __init__(self):
        pass

    def mask_ip_addresses(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        def mask(match):
            ip = match.group(0)
            try:
                ipaddress.ip_network(ip, strict=False)
                key = f"pos_{pos_counter[0]}"
                extracted[key] = ip
                pos_counter[0] += 1
                return '<IP>'
            except ValueError:
                return ip
        text = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}(?:/\d{1,2})?\b', mask, text)
        return text, extracted, pos_counter

    def mask_uuid(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        uuid_pattern = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b')
        def repl(match):
            key = f"pos_{pos_counter[0]}"
            extracted[key] = match.group(0)
            pos_counter[0] += 1
            return "<UUID>"
        return uuid_pattern.sub(repl, text), extracted, pos_counter

    def mask_timestamps(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        timestamp_patterns = [
            r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b',
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b'
        ]
        for p in timestamp_patterns:
            def repl(match):
                key = f"pos_{pos_counter[0]}"
                extracted[key] = match.group(0)
                pos_counter[0] += 1
                return "<TIME>"
            text = re.sub(p, repl, text)
        return text, extracted, pos_counter

    def mask_durations(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        def repl(match):
            val = match.group(0)
            key = f"pos_{pos_counter[0]}"
            extracted[key] = val
            pos_counter[0] += 1
            return "<DURATION>"
        return re.sub(r'\b\d+(?:\.\d+)?\s*(ms|msec|s|sec)\b', repl, text, flags=re.IGNORECASE), extracted, pos_counter

    def mask_alphanumeric_numbers(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        def repl(match):
            val = match.group(0)
            key = f"pos_{pos_counter[0]}"
            extracted[key] = val
            pos_counter[0] += 1
            # Mask numeric part only
            prefix, digits, suffix = match.group(1), match.group(2), match.group(3)
            return f"{prefix}<NUM>{suffix}"
        return re.sub(r'(?<!<)([A-Za-z]*)(\d+)([A-Za-z]*)', repl, text), extracted, pos_counter

    def mask_numbers(self, text: str, extracted: dict, pos_counter: int) -> tuple[str, dict, int]:
        def repl(match):
            key = f"pos_{pos_counter[0]}"
            extracted[key] = int(match.group(0))
            pos_counter[0] += 1
            return "<NUM>"
        return re.sub(r'\b(?<!<)\d+(?!>)\b', repl, text), extracted, pos_counter

    def clean(self, log_line: str):
        """Return masked log and positional dynamic value mapping."""
        if not log_line or len(log_line.strip()) < 5:
            return None, {}

        extracted = {}
        pos_counter = [0]
        log = log_line

        log, extracted, pos_counter = self.mask_timestamps(log, extracted, pos_counter)
        log, extracted, pos_counter = self.mask_ip_addresses(log, extracted, pos_counter)
        log, extracted, pos_counter = self.mask_uuid(log, extracted, pos_counter)
        log, extracted, pos_counter = self.mask_durations(log, extracted, pos_counter)
        log, extracted, pos_counter = self.mask_alphanumeric_numbers(log, extracted, pos_counter)
        log, extracted, pos_counter = self.mask_numbers(log, extracted, pos_counter)

        log = re.sub(r'\s+', ' ', log).strip()
        return log, extracted
