from src.preprocessor.modules.logProcessor import LogPreprocessor

def main():
    print("Hello from major-project!")
    sample_log="""2023-11-20T09:04:13.149781	DEBUG	ServiceB	File I/O	7147	User49	192.168.1.97	21ms
"""
    processor = LogPreprocessor()
    cleaned_log, extracted = processor.clean(sample_log)
    print(f"Original Log: {sample_log}")
    print(f"Cleaned Log: {cleaned_log}")
    print(f"Extracted Values: {extracted}")


if __name__ == "__main__":
    main()
