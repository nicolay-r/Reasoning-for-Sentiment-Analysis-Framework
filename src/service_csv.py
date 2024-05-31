import csv

from tqdm import tqdm


class CsvService:

    @staticmethod
    def write(target, lines_it):
        f = open(target, "w")
        print(f"Saving: {target}")
        w = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for content in lines_it:
            w.writerow(content)

    @staticmethod
    def write_handled(target, data_it, data2col_func, header):

        def __it():
            yield ["row_id"] + header
            for row_id, data in data_it:
                content = data2col_func(data)
                assert(len(content) == len(header))
                yield [row_id] + content

        CsvService.write(target, lines_it=__it())

    @staticmethod
    def read(target, skip_header=False, cols=None, as_dict=False, **csv_kwargs):
        assert (isinstance(cols, list) or cols is None)

        header = None
        with open(target, newline='\n') as f:
            for row_id, row in tqdm(enumerate(csv.reader(f, **csv_kwargs)), desc="Reading CSV"):
                if skip_header and row_id == 0:
                    header = row
                    continue

                # Determine the content we wish to return.
                if cols is None:
                    content = row
                else:
                    row_d = {header[col_name]: value for col_name, value in enumerate(row)}
                    content = [row_d[col_name] for col_name in cols]

                # Optionally attach row_id to the content.
                if as_dict:
                    assert (header is not None)
                    assert (len(content) == len(header))
                    yield {k: v for k, v in zip(header, content)}
                else:
                    yield content
