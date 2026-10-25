"""Template-based generator of synthetic, annotated LLM-chat prompts in four
language conditions (en, kk, ru, cs). Gold spans are recorded at generation
time (character offsets), so annotation is exact by construction.

All identifiers are synthetic: IINs and IBANs are random numbers that only
satisfy the published checksum rules; they do not belong to real people.

Usage:  python src/generate_prompts.py --n 60 --seed 42 --out data/prompts.jsonl
"""
import argparse, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from resources import FIRST_NAMES, SURNAMES, CITIES, STREETS, ORGS, EMAIL_DOMAINS

IIN_W1 = list(range(1, 12))
IIN_W2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]

def iin_check_digit(d11):
    s = sum(int(a) * w for a, w in zip(d11, IIN_W1)) % 11
    if s == 10:
        s = sum(int(a) * w for a, w in zip(d11, IIN_W2)) % 11
        if s == 10:
            return None
    return s

def make_iin(rng, gender):
    while True:
        year = rng.randint(1960, 2005)
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        if year < 2000:
            c = 3 if gender == "m" else 4
        else:
            c = 5 if gender == "m" else 6
        d11 = f"{year % 100:02d}{month:02d}{day:02d}{c}{rng.randint(0, 9999):04d}"
        cd = iin_check_digit(d11)
        if cd is not None:
            return d11 + str(cd)

def make_invalid_12(rng):
    """12-digit number that is NOT a valid IIN (hard negative: contract/order no.)."""
    while True:
        d = "".join(str(rng.randint(0, 9)) for _ in range(12))
        cd = iin_check_digit(d[:11])
        if cd is None or str(cd) != d[11]:
            return d

def iban_check(country, bban):
    s = bban + country + "00"
    num = "".join(str(int(ch, 36)) for ch in s)
    return 98 - int(num) % 97

def make_iban(rng):
    acc = "".join(str(rng.randint(0, 9)) for _ in range(13))
    if rng.random() < 0.5:
        acc = "KZT" + acc[3:]
    bban = f"{rng.randint(100, 999)}" + acc
    iban = f"KZ{iban_check('KZ', bban):02d}{bban}"
    if rng.random() < 0.4:
        iban = " ".join(iban[i:i + 4] for i in range(0, 20, 4))
    return iban

def make_phone(rng):
    op = rng.choice(["701", "702", "705", "707", "747", "771", "775", "777", "778"])
    a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
    fmt = rng.randint(0, 3)
    return [f"+7 {op} {a} {b:02d} {c:02d}", f"8{op}{a}{b:02d}{c:02d}",
            f"+7({op}){a}-{b:02d}-{c:02d}", f"8 {op} {a} {b:02d}{c:02d}"][fmt]

def make_person(rng):
    fn = rng.choice(FIRST_NAMES)
    sn = rng.choice(SURNAMES)
    return {"fn": fn, "sn": sn}

def person_surface(rng, p, script):
    cyr, lat, g, fn_in = p["fn"]
    s_cyr, s_lat, sn_in = p["sn"]
    first = cyr if script == "cyr" else lat
    last = s_cyr if script == "cyr" else s_lat
    if g == "f" and last not in ("Ким", "Kim"):
        last += "а" if script == "cyr" else "a"
    r = rng.random()
    if r < 0.55:
        form, text = "first_last", f"{first} {last}"
    elif r < 0.70:
        form, text = "last_first", f"{last} {first}"
    elif r < 0.85:
        form, text = "first_only", first
    elif r < 0.90:
        form, text = "last_only", last
    else:
        form, text = "lowercase", f"{first} {last}".lower()
    covered = (fn_in if form == "first_only" else sn_in if form == "last_only"
               else (fn_in or sn_in))
    return text, {"form": form, "in_gazetteer": covered}

def email_for(rng, p):
    lat, s_lat = p["fn"][1].lower(), p["sn"][1].lower()
    return f"{lat}.{s_lat}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"

T = {
 "en": [
  "Can you write a formal email to {ORG_EN} on behalf of my client {PER}? The client's IIN is {IIN} and the refund should go to account {IBAN}. Contact phone: {PHONE}.",
  "Summarize this complaint: My name is {PER}, I live at {ADDR_EN}, {CITY_EN}. On {DATE} I was charged {SUM} tenge twice. You can reach me at {EMAIL}.",
  "Translate into Russian: Dear {PER}, your application No. {ORDER} has been approved. Please visit our office in {CITY_EN} with your ID card.",
  "Fix the grammar: {PER} and {PER2} from {ORG_EN} will attends the meeting in {CITY_EN}, call {PHONE} if they are late.",
  "Draft a reference letter for my employee. Full name: {PER}, IIN {IIN}, working at {ORG_EN} since {YEAR}. Email: {EMAIL}.",
  "Make a table of these customers: {PER}, {PHONE}, {CITY_EN}; {PER2}, {EMAIL}, {CITY2_EN}.",
  "Check this payment order: transfer {SUM} KZT from {IBAN} to {PER}, contract No. {ORDER}.",
  "Write a short bio: {PER}, student at {ORG_EN}, lives in {CITY_EN}. Contacts: {EMAIL}, {PHONE}.",
 ],
 "ru": [
  "Напиши официальное письмо в {ORG_RU} от имени клиента. ФИО клиента: {PER}, ИИН {IIN}, возврат нужно сделать на счёт {IBAN}. Контактный телефон: {PHONE}.",
  "Кратко перескажи жалобу: Меня зовут {PER}, я живу по адресу {ADDR_RU}, г. {CITY_RU}. {DATE} с меня дважды списали {SUM} тенге. Мой e-mail: {EMAIL}.",
  "Переведи на английский: Уважаемый клиент {PER}, ваша заявка № {ORDER} одобрена. Пожалуйста, подойдите в наш офис в {CITY_RU_LOC} с удостоверением личности.",
  "Исправь ошибки: {PER} и {PER2} из {ORG_RU} будут на встрече в {CITY_RU_LOC}, если опаздают звоните {PHONE}.",
  "Составь характеристику для сотрудника. ФИО: {PER}, ИИН {IIN}, место работы: {ORG_RU} с {YEAR} года, почта: {EMAIL}.",
  "Сделай таблицу клиентов: {PER}, {PHONE}, {CITY_RU}; {PER2}, {EMAIL}, {CITY2_RU}.",
  "Проверь платёжное поручение: перевести {SUM} тенге со счёта {IBAN}, получатель {PER}, договор № {ORDER}.",
  "Напиши короткую биографию: {PER}, студент {ORG_RU}, город {CITY_RU}. Контакты: {EMAIL}, {PHONE}.",
 ],
 "kk": [
  "{ORG_KK} компаниясына клиент атынан ресми хат жазып бер. Клиенттің аты-жөні: {PER}, ЖСН: {IIN}. Ақшаны {IBAN} шотына қайтару керек. Байланыс телефоны: {PHONE}.",
  "Мына шағымды қысқаша мазмұнда: Менің атым {PER}, мен {CITY_KK} қаласы, {ADDR_KK} мекенжайында тұрамын. {DATE} күні менен екі рет {SUM} теңге алынды. Электрондық поштам: {EMAIL}.",
  "Ағылшын тіліне аудар: Құрметті {PER}, сіздің № {ORDER} өтінішіңіз мақұлданды. Жеке куәлігіңізбен {CITY_KK_LOC} орналасқан кеңсемізге келіңіз.",
  "Қателерін түзет: {PER} және {PER2} {ORG_KK} атынан {CITY_KK_LOC} өтетін кездесуге қатысады, кешігетін болса {PHONE} нөміріне хабарласыңыз.",
  "Қызметкерге мінездеме жазып бер. Аты-жөні: {PER}, ЖСН {IIN}, жұмыс орны: {YEAR} жылдан бері {ORG_KK}. Пошта: {EMAIL}.",
  "Клиенттер кестесін жаса: {PER}, {PHONE}, {CITY_KK}; {PER2}, {EMAIL}, {CITY2_KK}.",
  "Төлем тапсырмасын тексер: {IBAN} шотынан {PER} атына {SUM} теңге аудару, шарт № {ORDER}.",
  "Қысқаша өмірбаян жаз: {PER}, {ORG_KK} студенті, {CITY_KK} қаласы. Байланыс: {EMAIL}, {PHONE}.",
 ],
 "cs": [
  "Слушай, помоги письмо написать в {ORG_RU}. Клиенттің аты-жөні {PER}, ИИН-і {IIN}, деньги на счёт {IBAN} қайтару керек. Телефоны {PHONE}.",
  "Мына жалобаны кратко перескажи: менің атым {PER}, живу по адресу {ADDR_RU}, {CITY_RU}. {DATE} екі рет {SUM} тенге списали. Почтам: {EMAIL}.",
  "На английский аударып бер: Құрметті {PER}, ваша заявка № {ORDER} мақұлданды, приходите в офис в {CITY_RU_LOC}.",
  "Ошибкаларын түзетші: {PER} и {PER2} из {ORG_RU} ертең встречаға келеді {CITY_KK_LOC}, если что звоните {PHONE}.",
  "Сотрудникке характеристика жазып бер: ФИО {PER}, ЖСН {IIN}, {YEAR} жылдан бері {ORG_RU} работает. Почта: {EMAIL}.",
  "Таблица жаса клиенттерге: {PER}, {PHONE}, {CITY_RU}; {PER2}, {EMAIL}, {CITY2_RU}.",
  "Платёжканы тексерші: {IBAN} счётынан получатель {PER}, {SUM} тенге перевод, договор № {ORDER}.",
  "Короче, биография жаз: {PER}, {ORG_KK} студенті, город {CITY_RU}. Контакты: {EMAIL}, {PHONE}.",
 ],
}

def address(rng, lang):
    kk, ru, lat = rng.choice(STREETS)
    h, a = rng.randint(1, 180), rng.randint(1, 120)
    if lang == "en":
        return f"{h} {lat} Avenue, apt {a}"
    if lang == "kk":
        return f"{kk} көшесі, {h}-үй, {a}-пәтер"
    return f"ул. {ru}, д. {h}, кв. {a}"

SLOT = re.compile(r"\{(\w+)\}")

def fill(rng, lang, template, pid):
    script = "lat" if lang == "en" else "cyr"
    if lang == "cs" and rng.random() < 0.25:
        script = "lat"
    p1, p2 = make_person(rng), make_person(rng)
    while p2["fn"] == p1["fn"]:
        p2 = make_person(rng)
    c1, c2 = rng.sample(CITIES, 2)
    org = rng.choice(ORGS)
    vals = {}
    def v(slot):
        if slot in vals:
            return vals[slot]
        if slot in ("PER", "PER2"):
            p = p1 if slot == "PER" else p2
            t, meta = person_surface(rng, p, script)
            r = (t, "PERSON", meta)
        elif slot.startswith("CITY"):
            c = c1 if not slot.startswith("CITY2") else c2
            kn, kl, rn, rl, lat, inz = c
            form = {"CITY_EN": lat, "CITY2_EN": lat, "CITY_KK": kn, "CITY2_KK": kn,
                    "CITY_KK_LOC": kl, "CITY_RU": rn, "CITY2_RU": rn, "CITY_RU_LOC": rl}[slot]
            r = (form, "LOC", {"form": slot, "in_gazetteer": inz})
        elif slot.startswith("ADDR"):
            r = (address(rng, slot[-2:].lower()), "LOC", {"form": "address", "in_gazetteer": None})
        elif slot.startswith("ORG"):
            en, kk, ru, inz = org
            r = ({"ORG_EN": en, "ORG_KK": kk, "ORG_RU": ru}[slot], "ORG",
                 {"form": "legal_form" if "«" in kk else "name", "in_gazetteer": inz})
        elif slot == "IIN":
            r = (make_iin(rng, p1["fn"][2]), "IIN", {})
        elif slot == "IBAN":
            r = (make_iban(rng), "IBAN", {})
        elif slot == "PHONE":
            r = (make_phone(rng), "PHONE", {})
        elif slot == "EMAIL":
            r = (email_for(rng, p1), "EMAIL", {})
        elif slot == "ORDER":
            r = (make_invalid_12(rng) if rng.random() < 0.6 else str(rng.randint(100000, 9999999)), None, {})
        elif slot == "DATE":
            r = (f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2025", None, {})
        elif slot == "SUM":
            r = (f"{rng.randint(5, 950)} 000", None, {})
        elif slot == "YEAR":
            r = (str(rng.randint(2012, 2025)), None, {})
        else:
            raise KeyError(slot)
        vals[slot] = r
        return r

    text, spans, pos = "", [], 0
    for m in SLOT.finditer(template):
        text += template[pos:m.start()]
        surface, label, meta = v(m.group(1))
        if label:
            spans.append({"start": len(text), "end": len(text) + len(surface),
                          "label": label, "text": surface, **{f"meta_{k}": val for k, val in meta.items()}})
        text += surface
        pos = m.end()
    text += template[pos:]
    return {"id": pid, "lang": lang, "template": T[lang].index(template), "text": text, "entities": spans}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="prompts per language condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for lang in ["en", "kk", "ru", "cs"]:
        for i in range(a.n):
            tpl = T[lang][i % len(T[lang])]
            rows.append(fill(rng, lang, tpl, f"{lang}-{i:03d}"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts, {sum(len(r['entities']) for r in rows)} gold entities -> {a.out}")

if __name__ == "__main__":
    main()
