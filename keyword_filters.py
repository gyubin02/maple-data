from __future__ import annotations

from typing import Dict, Iterable, List


CATEGORY_SYNONYMS = {
    "모자": ["모자", "헬름", "헬멧", "햇", "보닛", "캡", "hat", "cap", "helmet"],
    "신발": ["신발", "슈즈", "부츠", "샌들", "shoes", "shoe", "boots", "sandal"],
    "장갑": ["장갑", "글러브", "glove", "gloves"],
    "무기": [
        "무기",
        "검",
        "소드",
        "대검",
        "스태프",
        "완드",
        "활",
        "석궁",
        "창",
        "스피어",
        "폴암",
        "도끼",
        "단검",
        "너클",
        "건",
        "총",
        "클로",
        "weapon",
        "sword",
        "staff",
        "wand",
        "bow",
        "spear",
        "axe",
        "dagger",
        "gun",
        "claw",
    ],
    "상의": [
        "상의",
        "셔츠",
        "자켓",
        "코트",
        "로브",
        "블라우스",
        "top",
        "shirt",
        "jacket",
        "coat",
        "robe",
        "blouse",
    ],
    "하의": ["하의", "바지", "팬츠", "스커트", "bottom", "pants", "skirt"],
    "망토": ["망토", "케이프", "cape", "날개", "윙", "wing", "wings"],
    "귀걸이": ["귀걸이", "귀고리", "이어링", "earring", "earrings"],
    "반지": ["반지", "링", "ring"],
    "목걸이": ["목걸이", "펜던트", "네클리스", "necklace", "pendant"],
    "벨트": ["벨트", "belt"],
    "얼굴장식": ["얼굴장식", "얼굴 장식"],
    "눈장식": ["눈장식", "눈 장식"],
    "보조무기": ["보조무기", "보조 무기", "sub weapon", "subweapon", "offhand"],
    "방패": ["방패", "쉴드", "실드", "shield"],
}

COLOR_SYNONYMS = {
    "black": ["검은", "검정", "블랙", "black"],
    "white": ["흰", "하얀", "화이트", "white"],
    "gray": ["회색", "그레이", "gray"],
    "silver": ["은색", "실버", "silver"],
    "gold": ["금색", "골드", "gold"],
    "red": ["빨간", "빨강", "레드", "적색", "붉은", "red"],
    "pink": ["핑크", "분홍", "분홍색", "핑크색", "pink"],
    "orange": ["주황", "오렌지", "orange"],
    "yellow": ["노란", "노랑", "옐로", "yellow"],
    "green": ["초록", "녹색", "그린", "green"],
    "blue": ["파란", "파랑", "블루", "blue", "하늘색", "스카이", "sky"],
    "purple": ["보라", "퍼플", "purple"],
    "brown": ["갈색", "브라운", "brown"],
    "beige": ["베이지", "beige"],
    "mint": ["민트", "mint"],
    "teal": ["청록", "teal", "터쿼이즈", "turquoise"],
    "navy": ["남색", "네이비", "navy"],
}

VIBE_SYNONYMS = {
    "cute": ["귀여움", "귀여운", "귀엽", "큐트", "cute", "사랑스러운", "lovely"],
    "sporty": ["스포티", "스포츠", "sporty", "sports", "스포티한"],
    "casual": ["캐주얼", "casual"],
    "luxury": ["고급스러움", "고급", "luxury", "classy", "품격", "vip", "VIP"],
    "elegant": ["우아", "elegant", "고상", "세련", "세련된"],
    "playful": ["유쾌한", "funny", "playful", "장난", "발랄"],
    "bright": ["빛나는", "sparkle", "glitter", "반짝", "sparkling"],
    "powerful": ["강력한", "전투적인", "전투용", "powerful", "강인"],
    "romantic": ["로맨틱", "romance", "romantic", "설렘", "사랑"],
    "mysterious": ["신비", "mysterious", "묘한"],
    "retro": ["레트로", "retro", "빈티지", "vintage", "클래식", "classic", "고전적인"],
    "futuristic": ["futuristic", "미래", "사이버", "sf"],
    "sweet": ["달달", "달콤", "sweet", "상큼"],
    "unique": ["유니크", "unique", "독특", "개성", "특별"],
    "calm": ["고요한", "차분", "calm"],
    "dark": ["다크", "dark", "ダーク", "어두운"],
}


def extract_keywords(
    texts: Iterable[str], synonyms: Dict[str, List[str]]
) -> List[str]:
    lowered_texts = [text.lower() for text in texts if text]
    if not lowered_texts:
        return []
    hits: List[str] = []
    for canonical, variants in synonyms.items():
        for variant in variants:
            variant_lower = variant.lower()
            if any(variant_lower in text for text in lowered_texts):
                hits.append(canonical)
                break
    return hits
