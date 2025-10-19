def make_demo_prompts(n: int):
    base = [
        "Q: 3人で24個のクッキーを等分します。1人あたり何個ですか？ A:",
        "Q: 列車は時速80kmで2.5時間走る。走行距離は？ A:",
        "Explain briefly why the sky appears blue.",
        "Summarize in one sentence: The importance of data structures.",
        "日本の都道府県を5つ挙げて。",
        "Write a Python function to check if a number is prime.",
        "Q: 2x + 7 = 19 の x を求めよ。A:",
        "Explain the concept of attention in transformers.",
        "この文を丁寧語に: それ、やって。",
        "Give three use-cases of KV caching in LLMs.",
    ]
    return (base * ((n + len(base) - 1)//len(base)))[:n]
