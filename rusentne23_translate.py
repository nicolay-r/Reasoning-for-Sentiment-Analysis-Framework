import argparse
from tqdm import tqdm
from os.path import join, basename, dirname

from src.service_csv import CsvService
from src.translator import translate_value


def iter_translated_rusentne2023(src, need_label=False, lang="en"):
    cols = ["sentence", "entity_tag", "entity_pos_start_rel", "entity_pos_end_rel"] \
           + (["label"] if need_label else [])
    it = CsvService.read(target=src, delimiter='\t', cols=cols, skip_header=True)

    # These are the list of the meta entries that are expected to be replaced
    # in order to prevent their corruption during translation process.
    # (Specifics of the GoogleTranslate backend engine)
    meta = [
        ("[META1]", "[META2]"),
        ("<{[", "]}>"),
        ("{[<", ">]}"),
        ("<e>", "</e>"),
        ("$$$", "###"),
        ("$$$", "&&&"),
        ("&&&", "###"),
        ("|||", "###"),
        ("||{|", "|}||"),
        ("{|", "|}")
    ]

    yield [cols[0], cols[1], "entity", cols[2], cols[3]] + ([cols[4]] if need_label else [])

    l_ind = 0
    for line_data in tqdm(it, desc=f"Translating to `{lang}` ({src})", unit="line"):

        text = line_data[0]
        e_tag = line_data[1]
        e_from = int(line_data[2])
        e_to = int(line_data[3])

        label = None
        if need_label:
            label = line_data[4]

        is_translated = False
        for P, S in meta:

            p = text[:e_from].strip()
            s = text[e_to:].strip()
            m = P + text[e_from:e_to] + S

            t = " ".join([p, m, s]).strip()

            t = translate_value(t, src="ru", dest=lang)

            if (P not in t) or (S not in t):
                continue

            new_from = t.index(P)
            new_to = t.index(S) - len(P)
            entity = t[new_from+len(P):t.index(S)]
            t = t.replace(P, "").replace(S, "")

            yield [t, e_tag, entity, new_from, new_to] + ([label] if label is not None else [])

            is_translated = True
            break

        if not is_translated:
            print("Skipped: {}".format(l_ind))

        l_ind += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Translate RuSentNE-2023 dataset")

    parser.add_argument('--src', dest='src', type=str)
    parser.add_argument('--lang', dest='lang', type=str, default="en")
    parser.add_argument('--label', dest='need_label', action='store_true', default=False)

    args = parser.parse_args()

    filename = '.'.join(basename(args.src).split('.')[:-1])
    target = join(dirname(args.src), f"{filename}_{args.lang}.csv")

    CsvService.write(target=target,
                     lines_it=iter_translated_rusentne2023(args.src, need_label=args.need_label, lang=args.lang))

