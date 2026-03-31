"""Rule-based A-share news event typing."""

import json


EVENT_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("earnings", ["业绩", "营收", "利润", "净利", "财报", "年报", "季报", "中报", "预增", "预减", "扭亏", "快报"]),
    ("policy", ["政策", "监管", "证监会", "国务院", "发改委", "工信部", "央行", "降准", "降息", "税收", "补贴"]),
    ("order_contract", ["订单", "中标", "合同", "签约", "项目", "供货", "采购", "交付"]),
    ("product_tech", ["产品", "技术", "研发", "芯片", "人工智能", "ai", "模型", "专利", "平台", "创新", "新能源"]),
    ("buyback_increase", ["回购", "增持", "员工持股", "股权激励"]),
    ("reduction_unlock", ["减持", "解禁", "套现", "清仓"]),
    ("mna_restructuring", ["并购", "重组", "收购", "资产重组", "借壳", "整合", "注入"]),
    ("litigation_penalty", ["诉讼", "仲裁", "处罚", "立案", "调查", "罚款", "违规", "问询函"]),
    ("management", ["董事长", "总经理", "高管", "董监高", "辞职", "离任", "任命", "管理层", "董事会"]),
]


def classify_event_types(*parts: str | None) -> list[str]:
    text = " ".join(str(part or "") for part in parts).lower()
    matched: list[str] = []
    for event_type, keywords in EVENT_TYPE_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            matched.append(event_type)

    if not matched:
        return ["other"]
    return matched


def event_types_to_json(event_types: list[str]) -> str:
    return json.dumps(event_types, ensure_ascii=False)


def parse_event_types(raw_event_types: str | None, *fallback_parts: str | None) -> list[str]:
    if raw_event_types:
        try:
            parsed = json.loads(raw_event_types)
            if isinstance(parsed, list):
                normalized = [str(item) for item in parsed if str(item).strip()]
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            pass
    return classify_event_types(*fallback_parts)
