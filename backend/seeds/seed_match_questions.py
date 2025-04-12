from recommender.models import MatchQuestion, MatchOption

questions = [
    {
        "id": "budget_weight",
        "title": "💰 Rent prices",
        "description": "How sensitive are you to rent prices?",
        "options": [
            {"label": "I'm on a tight budget – affordability is key", "value": 1},
            {"label": "I care about price, but I can stretch a bit", "value": 0.5},
            {"label": "I'm willing to pay more for a better area", "value": 0},
        ],
    },
    {
        "id": "safety_weight",
        "title": "🛡️ Safety",
        "description": "How much does safety influence your choice of area?",
        "options": [
            {"label": "I won't compromise on safety", "value": 1},
            {"label": "I'd like a safe area, but I'm flexible", "value": 0.5},
            {"label": "Not a big concern for me", "value": 0},
        ],
    },
    {
        "id": "centrality_weight",
        "title": "📍 Location",
        "description": "How important is it for you to live close to central London?",
        "options": [
            {"label": "I want to be in the heart of the city", "value": 1},
            {"label": "It’d be nice, but I’m flexible", "value": 0.5},
            {"label": "I don’t mind being further out", "value": 0},
        ],
    },
    {
        "id": "youth_weight",
        "title": "🎉 Atmosphere",
        "description": "What kind of neighbourhood vibe are you looking for?",
        "options": [
            {"label": "Energetic and youthful", "value": 1},
            {"label": "Calm and family-oriented", "value": 0.5},
            {"label": "I don’t mind either way", "value": 0},
        ],
    },
    {
        "id": "stay_duration",
        "title": "Duration",
        "description": "How long do you plan to stay?",
        "options": [
            {"label": "Less than a year", "value": "short_term"},
            {"label": "1–2 years", "value": "mid_term"},
            {"label": "Longer than 2 years", "value": "long_term"},
            {"label": "Not sure yet / Open to anything", "value": "unknown"},
        ],
    },
    {
        "id": "is_student",
        "title": "Current situation",
        "description": "What best describes your current situation?",
        "options": [
            {"label": "I'm a student", "value": "student"},
            {"label": "I'm a young professional", "value": "young-professional"},
            {"label": "I’m relocating with family", "value": "family"},
            {"label": "Other", "value": "other"},
        ],
    },
]

for q in questions:
    question, created = MatchQuestion.objects.get_or_create(
        id=q["id"],
        defaults={
            "title": q["title"],
            "description": q["description"],
            "question_type": "choice",
        },
    )

    if not created:
        # Update existing question info
        question.title = q["title"]
        question.description = q["description"]
        question.save()

    # Remove previous options
    question.options.all().delete()

    for idx, opt in enumerate(q["options"]):
        MatchOption.objects.create(
            question=question,
            label=opt["label"],
            value=str(opt["value"]),
            order=idx,
        )
