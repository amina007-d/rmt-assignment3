"""Shared resources: entity taxonomy, name/place pools used by the prompt generator,
and the (smaller) gazetteers used by the rule-based detector.

IMPORTANT (leakage control): the generator draws people, places and organisations
from a pool that is LARGER than the gazetteers. Items marked in_gaz=False are
deliberately absent from the rule-based gazetteers, so rule-based recall on
unstructured entities is not inflated by construction. Recall is reported
separately for covered vs. uncovered items.
"""

LABELS = ["PERSON", "LOC", "ORG", "IIN", "IBAN", "PHONE", "EMAIL"]
STRUCTURED = {"IIN", "IBAN", "PHONE", "EMAIL"}
UNSTRUCTURED = {"PERSON", "LOC", "ORG"}
LANGS = ["en", "kk", "ru", "cs"]

FIRST_NAMES = [
    ("Айгерім", "Aigerim", "f", True), ("Ерлан", "Erlan", "m", True),
    ("Данияр", "Daniyar", "m", True), ("Әсел", "Assel", "f", True),
    ("Нұрлан", "Nurlan", "m", True), ("Мадина", "Madina", "f", True),
    ("Арман", "Arman", "m", True), ("Жанар", "Zhanar", "f", True),
    ("Иван", "Ivan", "m", True), ("Елена", "Elena", "f", True),
    ("Сергей", "Sergey", "m", True), ("Ольга", "Olga", "f", True),

    ("Ақбота", "Akbota", "f", False), ("Бекзат", "Bekzat", "m", False),
    ("Толғанай", "Tolganai", "f", False), ("Әлихан", "Alikhan", "m", False),
    ("Гүлназ", "Gulnaz", "f", False), ("Дархан", "Darkhan", "m", False),
    ("Виктория", "Viktoriya", "f", False), ("Тимур", "Timur", "m", False),
    ("Сымбат", "Symbat", "f", False), ("Мирас", "Miras", "m", False),
]

SURNAMES = [
    ("Ахметов", "Akhmetov", True), ("Жұмабаев", "Zhumabayev", True),
    ("Иванов", "Ivanov", True), ("Петров", "Petrov", True),
    ("Смағұлов", "Smagulov", True), ("Ким", "Kim", True),
    # not in gazetteer
    ("Сәрсенбаев", "Sarsenbayev", False), ("Қасымов", "Kassymov", False),
    ("Тоқтаров", "Toktarov", False), ("Кузнецов", "Kuznetsov", False),
    ("Есенов", "Yessenov", False), ("Әбдірахманов", "Abdirakhmanov", False),
]

CITIES = [
    ("Алматы", "Алматыда", "Алматы", "Алматы", "Almaty", True),
    ("Астана", "Астанада", "Астана", "Астане", "Astana", True),
    ("Шымкент", "Шымкентте", "Шымкент", "Шымкенте", "Shymkent", True),
    ("Қарағанды", "Қарағандыда", "Караганда", "Караганде", "Karaganda", True),
    ("Ақтөбе", "Ақтөбеде", "Актобе", "Актобе", "Aktobe", True),
    ("Атырау", "Атырауда", "Атырау", "Атырау", "Atyrau", True),
    ("Павлодар", "Павлодарда", "Павлодар", "Павлодаре", "Pavlodar", True),
    ("Түркістан", "Түркістанда", "Туркестан", "Туркестане", "Turkistan", True),
    ("Қостанай", "Қостанайда", "Костанай", "Костанае", "Kostanay", False),
    ("Орал", "Оралда", "Уральск", "Уральске", "Oral", False),
    ("Өскемен", "Өскеменде", "Усть-Каменогорск", "Усть-Каменогорске", "Oskemen", False),
    ("Талдықорған", "Талдықорғанда", "Талдыкорган", "Талдыкоргане", "Taldykorgan", False),
]

STREETS = [("Абай", "Абая", "Abay"), ("Сәтбаев", "Сатпаева", "Satpayev"),
           ("Төле би", "Толе би", "Tole Bi"), ("Кенесары", "Кенесары", "Kenesary"),
           ("Тұран", "Туран", "Turan")]

ORGS = [
    ("Kaspi Bank", "Kaspi Bank", "Kaspi Bank", True),
    ("Halyk Bank", "Halyk Bank", "Halyk Bank", True),
    ("Kazakhtelecom", "Қазақтелеком", "Казахтелеком", True),
    ("Nur Kurylys LLP", "«Нұр Құрылыс» ЖШС", "ТОО «Нур Курылыс»", True),
    ("Altyn Dala JSC", "«Алтын Дала» АҚ", "АО «Алтын Дала»", True),
    ("Freedom Bank", "Freedom Bank", "Freedom Bank", False),
    ("Astana IT University", "Astana IT University", "Astana IT University", False),
    ("Air Astana", "Air Astana", "Air Astana", False),
]

EMAIL_DOMAINS = ["gmail.com", "mail.ru", "inbox.kz", "astanait.edu.kz"]

def gaz_first_names():
    out = set()
    for c, l, g, inz in FIRST_NAMES:
        if inz:
            out.update({c, l})
    return out

def gaz_surname_stems():
    out = set()
    for c, l, inz in SURNAMES:
        if inz:
            out.update({c, l})
    return out

def gaz_cities():
    out = set()
    for kn, kl, rn, rl, lat, inz in CITIES:
        if inz:
            out.update({kn, rn, lat})
    return out

def gaz_orgs():
    out = set()
    for en, kk, ru, inz in ORGS:
        if inz and "«" not in kk:
            out.update({en, kk, ru})
    return out