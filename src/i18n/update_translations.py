import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "de": {
        "adjustments": {
            "effects": {
                "amount": "Stärke",
                "bokehCircular": "Kreisförmig",
                "bokehHexagonal": "Sechseckig",
                "bokehOctagonal": "Achteckig",
                "bokehRing": "Blase (Ring)",
                "bokehShape": "Bokeh-Form",
                "lensBlur": "Objektivunschärfe",
                "lensDiffusion": "Streuung"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "KI-Tiefenkarte wird berechnet..."
            }
        }
    },
    "en": {
        "adjustments": {
            "effects": {
                "amount": "Amount",
                "bokehCircular": "Circular",
                "bokehHexagonal": "Hexagonal",
                "bokehOctagonal": "Octagonal",
                "bokehRing": "Bubble (Ring)",
                "bokehShape": "Bokeh Shape",
                "lensBlur": "Lens Blur",
                "lensDiffusion": "Diffusion"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Calculating AI Depth Map..."
            }
        }
    },
    "es": {
        "adjustments": {
            "effects": {
                "amount": "Cantidad",
                "bokehCircular": "Circular",
                "bokehHexagonal": "Hexagonal",
                "bokehOctagonal": "Octogonal",
                "bokehRing": "Burbuja (Anillo)",
                "bokehShape": "Forma del bokeh",
                "lensBlur": "Desenfoque de lente",
                "lensDiffusion": "Difusión"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Calculando mapa de profundidad IA..."
            }
        }
    },
    "fr": {
        "adjustments": {
            "effects": {
                "amount": "Intensité",
                "bokehCircular": "Circulaire",
                "bokehHexagonal": "Hexagonal",
                "bokehOctagonal": "Octogonal",
                "bokehRing": "Bulle (Anneau)",
                "bokehShape": "Forme du bokeh",
                "lensBlur": "Flou d'objectif",
                "lensDiffusion": "Diffusion"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Calcul de la carte de profondeur IA..."
            }
        }
    },
    "it": {
        "adjustments": {
            "effects": {
                "amount": "Quantità",
                "bokehCircular": "Circolare",
                "bokehHexagonal": "Esagonale",
                "bokehOctagonal": "Ottagonale",
                "bokehRing": "Bolla (Anello)",
                "bokehShape": "Forma del bokeh",
                "lensBlur": "Sfocatura obiettivo",
                "lensDiffusion": "Diffusione"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Calcolo mappa di profondità IA..."
            }
        }
    },
    "ja": {
        "adjustments": {
            "effects": {
                "amount": "適用量",
                "bokehCircular": "円形",
                "bokehHexagonal": "六角形",
                "bokehOctagonal": "八角形",
                "bokehRing": "バブル（リング）",
                "bokehShape": "ボケの形状",
                "lensBlur": "レンズぼかし",
                "lensDiffusion": "拡散"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "AI深度マップを計算中..."
            }
        }
    },
    "ko": {
        "adjustments": {
            "effects": {
                "amount": "양",
                "bokehCircular": "원형",
                "bokehHexagonal": "육각형",
                "bokehOctagonal": "팔각형",
                "bokehRing": "버블 (링)",
                "bokehShape": "보케 모양",
                "lensBlur": "렌즈 블러",
                "lensDiffusion": "확산"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "AI 심도 맵 계산 중..."
            }
        }
    },
    "pl": {
        "adjustments": {
            "effects": {
                "amount": "Ilość",
                "bokehCircular": "Kołowy",
                "bokehHexagonal": "Sześciokątny",
                "bokehOctagonal": "Ośmiokątny",
                "bokehRing": "Bąbelek (Pierścień)",
                "bokehShape": "Kształt bokeh",
                "lensBlur": "Rozmycie obiektywu",
                "lensDiffusion": "Dyfuzja"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Obliczanie mapy głębi AI..."
            }
        }
    },
    "pt": {
        "adjustments": {
            "effects": {
                "amount": "Quantidade",
                "bokehCircular": "Circular",
                "bokehHexagonal": "Hexagonal",
                "bokehOctagonal": "Octogonal",
                "bokehRing": "Bolha (Anel)",
                "bokehShape": "Forma do bokeh",
                "lensBlur": "Desfocagem de lente",
                "lensDiffusion": "Difusão"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "A calcular mapa de profundidade IA..."
            }
        }
    },
    "ru": {
        "adjustments": {
            "effects": {
                "amount": "Величина",
                "bokehCircular": "Круглая",
                "bokehHexagonal": "Шестиугольная",
                "bokehOctagonal": "Восьмиугольная",
                "bokehRing": "Пузырь (Кольцо)",
                "bokehShape": "Форма боке",
                "lensBlur": "Размытие объектива",
                "lensDiffusion": "Диффузия"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "Расчет карты глубины ИИ..."
            }
        }
    },
    "zh-CN": {
        "adjustments": {
            "effects": {
                "amount": "数量",
                "bokehCircular": "圆形",
                "bokehHexagonal": "六边形",
                "bokehOctagonal": "八边形",
                "bokehRing": "气泡（环形）",
                "bokehShape": "散景形状",
                "lensBlur": "镜头模糊",
                "lensDiffusion": "扩散"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "正在计算 AI 深度图..."
            }
        }
    },
    "zh-TW": {
        "adjustments": {
            "effects": {
                "amount": "數量",
                "bokehCircular": "圓形",
                "bokehHexagonal": "六角形",
                "bokehOctagonal": "八角形",
                "bokehRing": "氣泡（環形）",
                "bokehShape": "散景形狀",
                "lensBlur": "鏡頭模糊",
                "lensDiffusion": "擴散"
            }
        },
        "editor": {
            "ai": {
                "generatingDepthMap": "正在計算 AI 深度圖..."
            }
        }
    }
}

def deep_merge(target: dict, source: dict):
    """Recursively merges source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if isinstance(node, dict):
                deep_merge(node, value)
        else:
            target[key] = value

def sort_dict_recursively(item):
    if isinstance(item, dict):
        return {k: sort_dict_recursively(v) for k, v in sorted(item.items())}
    elif isinstance(item, list):
        return [sort_dict_recursively(x) for x in item]
    return item

def update_json_file(file_path: Path, trans: dict):
    if not file_path.exists():
        print(f"Skipping: {file_path.name} (File not found)")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {file_path.name}. Skipping.")
        return

    deep_merge(data, trans)
    sorted_data = sort_dict_recursively(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated and Sorted: {file_path.name}")

def main():
    if not LOCALES_DIR.exists():
        print(f"Error: Locales directory '{LOCALES_DIR}' does not exist.")
        return

    print("Starting lens blur translation updates...")
    for lang, trans in TRANSLATIONS.items():
        file_path = LOCALES_DIR / f"{lang}.json"
        update_json_file(file_path, trans)
    print("Done!")

if __name__ == "__main__":
    main()
