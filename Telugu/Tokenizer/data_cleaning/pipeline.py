from Telugu.Tokenizer.data_cleaning.basic_filer import BasicCorpusFilter
from Telugu.Tokenizer.data_cleaning.exact_dedup import ExactDeduplicator
from Telugu.Tokenizer.data_cleaning.quality_filter import AdvancedQualityFilter
from Telugu.Tokenizer.data_cleaning.near_dedup import NearDeduplicator


class Pipeline:
    def __init__(self, input_file, output_file, basic_filter_output=None, exact_dedup_output=None, quality_filter_output=None):
        self.input_file = input_file
        self.output_file = output_file

        self.basic_filter_output = basic_filter_output
        self.exact_dedup_output = exact_dedup_output
        self.quality_filter_output = quality_filter_output

    def run(self):
        # Step 1: Basic Corpus Filtering
        basic_filter = BasicCorpusFilter(self.input_file, self.basic_filter_output, min_chars=20, min_telugu_ratio =0.70)
        basic_filter.process_file()

        # Step 2: Exact Deduplication
        exact_dedup = ExactDeduplicator(self.basic_filter_output, self.exact_dedup_output, stats_file="exact_dedup_stats.json")
        exact_dedup.process_file()

        # Step 3: Advanced Quality Filtering
        quality_filter = AdvancedQualityFilter(self.exact_dedup_output, self.quality_filter_output, stats_file="quality_filter_stats.json")
        quality_filter.process_file()

        # Step 4: Near Deduplication
        near_dedup = NearDeduplicator(self.quality_filter_output, self.output_file, stats_file="near_dedup_stats.json")
        near_dedup.run()


if __name__ == "__main__":
    input_file = "Telugu/data/telugu_wikisource_raw.jsonl"
    output_file = "Telugu/data/telugu_wikisource_cleaned.jsonl"

    basic_filter_output = "Telugu/data/telugu_wikisource_basic_filtered.jsonl"
    exact_dedup_output = "Telugu/data/telugu_wikisource_exact_dedup.jsonl"
    quality_filter_output = "Telugu/data/telugu_wikisource_quality_filtered.jsonl"

    pipeline = Pipeline(
        input_file=input_file,
        output_file=output_file,
        basic_filter_output=basic_filter_output,
        exact_dedup_output=exact_dedup_output,
        quality_filter_output=quality_filter_output
    )
    pipeline.run()