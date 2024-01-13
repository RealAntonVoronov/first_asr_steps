# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    splitted_target = target_text.split()
    len_target = max(1e-5, len(splitted_target))
    return editdistance.distance(splitted_target, predicted_text.split()) / len_target


def calc_wer(target_text, predicted_text) -> float:
    len_target = max(1e-5, len(target_text))
    return editdistance.distance(target_text, predicted_text) / len_target