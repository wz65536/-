from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MAP_DIR = BASE_DIR / "province_map_data"

PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "香港",
    "澳门", "台湾",
]

CITY_PROVINCE = {name: name for name in PROVINCES}

FALLBACK_LABELS = ["自然风光", "城市地标", "历史文化", "休闲娱乐"]


CITY_ALIASES = {
    "北京": "北京",
    "天津": "天津",
    "上海": "上海",
    "重庆": "重庆",
    "福州": "福州",
    "厦门": "厦门",
    "泉州": "泉州",
    "漳州": "漳州",
    "莆田": "莆田",
    "三明": "三明",
    "南平": "南平",
    "龙岩": "龙岩",
    "宁德": "宁德",
}


@dataclass
class Attraction:
    province: str
    name: str
    location: str
    level: str
    heat: float
    reputation: str
    tags: str
    reviews: str
    text: str = ""
    label: str = ""
    label_source: str = ""
    score: float = 0.0


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MAP_DIR.mkdir(exist_ok=True)


def extract_city(location: str, province: str) -> str:
    location = str(location or "").strip()
    if not location or location == "#":
        return province
    city = location.split("·", 1)[0].strip()
    city = city.split("/", 1)[0].strip()
    city = city.split()[0].strip()
    if city in CITY_ALIASES:
        return CITY_ALIASES[city]
    if city in {"上海", "北京", "天津", "重庆"}:
        return city
    if city.endswith(("市", "州", "盟", "地区", "自治州", "自治县")):
        return city
    return city or province


def parse_attraction_file(file_path: Path, province: str) -> List[Attraction]:
    df = pd.read_csv(file_path, sep=";", dtype=str, keep_default_na=False)
    attractions: List[Attraction] = []
    for _, row in df.iterrows():
        name = str(row.get("景点名称", "")).strip()
        if not name or name.startswith("-") or name.startswith("="):
            continue
        location = str(row.get("地点", "")).strip()
        level = str(row.get("景点级别", "")).strip()
        heat_raw = str(row.get("热度", "")).strip()
        reputation = str(row.get("口碑", "")).strip()
        tags = str(row.get("标签", "")).strip()
        reviews = str(row.get("评论", "")).strip()
        heat = pd.to_numeric(pd.Series([heat_raw]), errors="coerce").fillna(0).iloc[0]
        city = extract_city(location, province)
        text = " ".join([name, city, location, level, reputation, tags, reviews])
        attractions.append(
            Attraction(
                province=province,
                name=name,
                location=location,
                level=level,
                heat=float(heat),
                reputation=reputation,
                tags=tags,
                reviews=reviews,
                text=text,
            )
        )
    return attractions


def clean_text(text: str) -> str:
    text = re.sub(r"[#\s]+", " ", text or "")
    text = re.sub(r"[（）()【】\[\]，,。；;:：/\\|]", " ", text)
    return text.strip().lower()


def create_training_data(attractions: List[Attraction]) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for item in attractions:
        labels.append(classify_by_fallback(item))
        texts.append(clean_text(build_tag_corpus(item)))
    return texts, labels


def build_model(texts: List[str], labels: List[str]) -> Optional[Pipeline]:
    if len(set(labels)) < 2 or len(texts) < 8:
        return None
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=12000)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto")),
        ]
    )
    model.fit(texts, labels)
    return model


def predict_label(item: Attraction, model: Optional[Pipeline]) -> Tuple[str, float, str]:
    if model is not None:
        probs = model.predict_proba([clean_text(build_tag_corpus(item))])[0]
        idx = int(np.argmax(probs))
        label = model.classes_[idx]
        prob = float(probs[idx])
        return str(label), prob, "ml"

    return classify_by_fallback(item), 0.35, "fallback"


def classify_by_fallback(item: Attraction) -> str:
    text = clean_text(build_tag_corpus(item))
    if any(k in text for k in ["山", "湖", "海", "峡谷", "森林", "瀑布", "草原", "湿地", "海岛"]):
        return "自然风光"
    if any(k in text for k in ["广场", "大厦", "观景", "步行街", "地标", "城区", "citywalk"]):
        return "城市地标"
    if any(k in text for k in ["古", "遗址", "宫", "庙", "寺", "城", "府", "馆", "文化"]):
        return "历史文化"
    if any(k in text for k in ["亲子", "乐园", "动物园", "海洋", "水族馆", "天文馆", "科技馆", "研学", "互动", "游乐"]):
        return "休闲娱乐"
    return "休闲娱乐"


def build_tag_corpus(item: Attraction) -> str:
    parts = [item.name, item.location, item.level, item.reputation, item.tags, item.reviews]
    return clean_text(" ".join([p for p in parts if p and p != "#"]))


def cluster_province_tags(attractions: List[Attraction]) -> Dict:
    if not attractions:
        return {"cluster_count": 0, "clusters": []}
    corpus = [build_tag_corpus(a) for a in attractions]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=6000)
    X = vectorizer.fit_transform(corpus)
    n_samples = X.shape[0]
    if n_samples < 3:
        cluster_ids = np.zeros(n_samples, dtype=int)
    else:
        n_clusters = min(8, max(2, int(round(np.sqrt(n_samples / 2)))))
        n_clusters = min(n_clusters, n_samples)
        if n_clusters < 2:
            cluster_ids = np.zeros(n_samples, dtype=int)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = model.fit_predict(X)
    clusters = []
    for cid in sorted(set(cluster_ids)):
        items = [a for a, label in zip(attractions, cluster_ids) if label == cid]
        top_tags = Counter()
        for a in items:
            for tag in re.split(r"[，,、\s]+", a.tags or ""):
                tag = tag.strip()
                if tag and tag != "#":
                    top_tags[tag] += 1
        clusters.append({
            "cluster_id": int(cid),
            "attraction_count": len(items),
            "top_tags": [name for name, _ in top_tags.most_common(8)],
            "representative_spots": [
                {
                    "name": a.name,
                    "location": a.location,
                    "level": a.level,
                    "tags": a.tags,
                    "label": a.label,
                    "score": round(a.score, 3),
                }
                for a in sorted(items, key=lambda x: (x.score, x.heat), reverse=True)[:10]
            ],
        })
    return {"cluster_count": len(clusters), "clusters": clusters}


def extract_city(location: str, province: str) -> str:
    location = (location or "").strip()
    if not location:
        return province
    city_part = location.split("·", 1)[0].strip()
    if not city_part or city_part == province:
        return province
    city_part = re.sub(r"[\s\-/（）()【】\[\]，,。；;:：]+.*$", "", city_part).strip()
    if city_part in {"北京", "天津", "上海", "重庆"}:
        return city_part
    return city_part


def aggregate_province(attractions: List[Attraction]) -> Dict:
    total = len(attractions)
    high_rated_count = sum(1 for a in attractions if a.level in {"4A", "5A"})
    label_counts = Counter(a.label for a in attractions)
    city_counts = Counter(
        extract_city(a.location, a.province) for a in attractions if a.level in {"4A", "5A"}
    )
    top_spots = sorted(attractions, key=lambda x: (x.score, x.heat), reverse=True)[:8]
    return {
        "province": attractions[0].province if attractions else "",
        "total_attractions": total,
        "high_rated_count": high_rated_count,
        "label_counts": dict(label_counts),
        "city_counts": dict(city_counts),
        "clusters": cluster_province_tags(attractions),
        "labels": [
            {"name": name, "value": count} for name, count in label_counts.items()
        ],
        "top_spots": [
            {
                "name": a.name,
                "location": a.location,
                "level": a.level,
                "heat": a.heat,
                "tags": a.tags,
                "label": a.label,
                "score": round(a.score, 3),
            }
            for a in top_spots
        ],
        "spots": [
            {
                "name": a.name,
                "location": a.location,
                "level": a.level,
                "heat": a.heat,
                "reputation": a.reputation,
                "tags": a.tags,
                "reviews": a.reviews,
                "label": a.label,
                "score": round(a.score, 3),
                "source": a.label_source,
            }
            for a in sorted(attractions, key=lambda x: (x.score, x.heat), reverse=True)
        ],
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_dirs()

    all_attractions: List[Attraction] = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        province = file_path.stem
        if province not in CITY_PROVINCE:
            continue
        if province == "全国景点":
            continue
        all_attractions.extend(parse_attraction_file(file_path, province))

    train_texts, train_labels = create_training_data(all_attractions)
    model = build_model(train_texts, train_labels)

    by_province: Dict[str, List[Attraction]] = defaultdict(list)
    for item in all_attractions:
        label, score, source = predict_label(item, model)
        item.label = label
        item.score = score
        item.label_source = source
        by_province[item.province].append(item)

    province_summary = {}
    national_labels = Counter()
    for province in PROVINCES:
        items = by_province.get(province, [])
        summary = aggregate_province(items)
        province_summary[province] = summary
        national_labels.update(summary["label_counts"])
        save_json(OUTPUT_DIR / "province" / f"{province}.json", summary)

    national = {
        "total_provinces": len([p for p in PROVINCES if by_province.get(p)]),
        "total_attractions": len(all_attractions),
        "label_counts": dict(national_labels),
        "provinces": [
            {
                "name": province,
                "total_attractions": province_summary[province]["total_attractions"],
                "high_rated_count": province_summary[province]["high_rated_count"],
                "label_counts": province_summary[province]["label_counts"],
            }
            for province in PROVINCES
        ],
    }

    save_json(OUTPUT_DIR / "national_summary.json", national)
    save_json(OUTPUT_DIR / "province_summary.json", province_summary)

    print(f"完成处理 {len(all_attractions)} 条景点数据")
    print(f"已生成全国汇总与 {len(province_summary)} 个省份的分类结果")


if __name__ == "__main__":
    main()
