import zipfile


class RuSentNE2023CodalabService:

    @staticmethod
    def read_submission(filepath, filename="baseline_results.txt"):
        archive = zipfile.ZipFile(filepath, 'r')
        with archive.open(filename) as f:
            return [int(line.strip()) for line in f.readlines()]
