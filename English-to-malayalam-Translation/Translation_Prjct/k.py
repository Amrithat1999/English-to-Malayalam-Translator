from transformers import MBartForConditionalGeneration, MBart50Tokenizer

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
tokenizer = MBart50Tokenizer.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)

# ---------- Hindi → Malayalam ----------
encoded_hi = tokenizer(
    article_hi,
    return_tensors="pt",
    src_lang="hi_IN"
)

generated_ml = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["ml_IN"],
    max_length=128
)

hi_to_ml = tokenizer.batch_decode(generated_ml, skip_special_tokens=True)
print("Hindi → Malayalam:")
print(hi_to_ml[0])

# ---------- Arabic → English ----------
encoded_ar = tokenizer(
    article_ar,
    return_tensors="pt",
    src_lang="ar_AR"
)

generated_en = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
    max_length=128
)

ar_to_en = tokenizer.batch_decode(generated_en, skip_special_tokens=True)
print("\nArabic → English:")
print(ar_to_en[0])
