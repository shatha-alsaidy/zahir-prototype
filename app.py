import os
from pathlib import Path
FARASA_HOME = Path("/tmp/farasa_home")
FARASA_HOME.mkdir(exist_ok=True, parents=True)
import farasa
farasa.base.FarasaBase.base_dir = FARASA_HOME
farasa.base.FarasaBase.bin_dir  = FARASA_HOME / "farasa_bin"
import torch
torch.classes.__path__ = []
import streamlit as st
import random
from farasa.pos import FarasaPOSTagger
from farasa.segmenter import FarasaSegmenter
from collections import Counter
import itertools
import numpy as np
import pandas as pd
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from transformers import pipeline
from datasets import load_dataset, interleave_datasets
import itertools
import re
import pickle

# Load precomputed label embeddings and zero-shot classifier ---
@st.cache_resource
def load_models():
    # load embeddings for labels
    with open('label_prototypes2.pkl', 'rb') as f:
        label_prototypes = pickle.load(f)

    # zero-shot classification pipeline
    zs = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')

    # embedding model
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large')

    # Load farasa pos tagger and segmenter
    farasa_pos = FarasaPOSTagger(interactive=True)
    farasa_segmenter = FarasaSegmenter(interactive=True)

    # Define the mOSCAR Arabic configs
    configs = ["arb_Arab", "ars_Arab", "ajp_Arab", "apc_Arab", "arz_Arab"]

    # Load each in streaming mode
    streams = []
    for cfg in configs:
        ds = load_dataset(
            "oscar-corpus/mOSCAR",
            cfg,
            split="train",
            streaming=True
        )
        # remove everything except the 'text' column
        ds = ds.remove_columns([col for col in ds.features if col != "text"])
        streams.append(ds)

    # Interleave them into one IterableDataset
    combined = interleave_datasets(streams)
    return label_prototypes, zs, embed_model, combined, farasa_pos, farasa_segmenter

label_prototypes, zs_classifier, embedding_model, combined, farasa_pos, farasa_segmenter = load_models()
labels = list(label_prototypes.keys())

# Synthetic test dataset 
@st.cache_data
def load_synthetic():
    dataset = {
    "شريكات": {
        "label": "مخدرات",
        "contexts": [
            "جربت شريكات البارح في الحفلة وكانت قوية مرة، خلت الكل يطير من الفرحة وما حسينا بحد للتعب.",
            "إذا تبي شريكات نظيفة بدون خلطات، تواصل معي على الخاص وخلك متأكد من الجودة.",
            "الدفعة الجديدة من شريكات اليوم مسكّت سعرها غالي، بس تستاهل عشان نقيّة جدًا.",
            "لا تنشر موضوع الشريكات في القروب العام، خلها بالخاص مع الموثوقين بس.",
            "شريكات الحمراء اللي جابوها الشباب كانت موسمية ولازم تحفظها في ثلاجة عشان ما تروح."  
        ]
    },
    "بيور": {
        "label": "مخدرات",
        "contexts": [
            "وصلت البيور اليوم بكرتون، نقاوتها عالية جدًا وبتعيش معك لأكثر من 12 ساعة.",
            "إذا حبيت تجرب البيور، اطلبها بدفع مسبق وخلك متواجد وقت التسليم.",
            "البير هذا مو مثل العسل، تأثيره ثقيل ويمشيك في عالم ثاني.",
            "ما ينصح بمزج البيور مع أي شي ثاني، بيور لوحده يعطيك المطلوب.",
            "جربنا البيور على فترتين وصار عندنا تقييم إيجابي من كل الناس."  
        ]
    },
    "ليمون": {
        "label": "غسيل اموال",
        "contexts": [
            "عندي ليمون سريع التحويل، يصل المبلغ لبنك الرياض خلال ساعة بدون أي شكوى.",
            "طريقة اللّيمون هذي مطوّرة مؤخرًا، ولا يترك أثر في أي تحقيق مالي.",
            "إذا تبي ليمون لدفع رواتب الشركة، أعطنا تفاصيل الحسابات والأسماء.",
            "خدمة اللّيمون الجزئي متوفرة بسعر منافس ويمكنك تقسيم المبلغ على دفعات.",
            "ليمون دبي صار يشتغل بواجهة جديدة ما تحتاج وكيل ولا وسيط."  
        ]
    },
    "كيو": {
        "label": "اختراق",
        "contexts": [
            "الكيو الجديدة تفتح حسابات البنوك الإلكترونية بدون ما يطلبون تحقق ثانوي.",
            "جربت كيو الـSSH على السيرفر الأساسي وكانت الاستجابة سريعة جداً.",
            "إذا تحتاج كيو فتح كاميرات المراقبة، أعطنا الموقع والوقت للترتيب.",
            "الكيو البنفسجية تضمن لك وصول صامت للنظام بدون أي بصمة رقمية.",
            "نبي نفك تشفير قاعدة البيانات، وقلت لهم الكيو الجاية بتنهي الشغل."  
        ]
    },
    "صفوة": {
        "label": "ارهاب",
        "contexts": [
            "اجتماع الصفوة راح يكون بعد الفجر في الخيمة الواقعة شرق الحديقة الكبيرة.",
            "الصفوة أشخاص متعاهدين، وأي تسريب للمعلومات يعد خيانة كبرى.",
            "تم توزيع المهام على أعضاء الصفوة وكان التنسيق ممتاز بدون خطأ.",
            "إذا وصلك رابط الصفوة، لا تشاركه مع أحد ولا تفتح عليه الجوال العام.",
            "هجمات الصفوة الماضية تمت بدقة عالية وبدون خسائر في المدنيين."  
        ]
    },
    "كاشير": {
        "label": "احتيال",
        "contexts": [
            "عندي كاشير فاخر يطلع لك شيك مزور يمر من أي صراف آلي بدون توقيف.",
            "طريقة الكاشير هذي تعتمد على ختم بنكي رقمي ما يكشفه النظام.",
            "جربت كاشير السوق السعودي وكانت العملية سلسة بدون رصيد ظاهر.",
            "لا تستخدم الكاشير مع تحويلات كبيرة بدون مراسلة خاصة أول.",
            "كاشير الخليج صار أسرع بعد التحديث الأخير وما يحتاج VPN."  
        ]
    },
    "دبس": {
        "label": "غسيل اموال",
        "contexts": [
            "دبس الـ500 ريال يطلع لك ورقة نظيفة كأنها جديدة من البنك مباشرة.",
            "إذا تبي دبس متعدد الألوان، يضمن لك تمرير المبالغ في عدة بنوك.",
            "طريقة الدبس السرية ما تعتمد على أي ختم مطبوع، فقط علامة مخفية.",
            "الدبس الفاخر يمشي لك حتى عند الصرافين الشعبيين بدون سؤال.",
            "شغل الدبس الجديد مرّن وعليه ضمانة شهر كامل لأي خلل."  
        ]
    },
    "فرش": {
        "label": "اسلحة",
        "contexts": [
            "وصلت الفرش الهجومية اليوم مع مخازن إضافية ورصاص عالي الجودة.",
            "فرش الستيليث يسمح لك بالدخول الخفي لأي مبنى بدون صوت.",
            "إذا بغيت فرش ثقيل للاستهداف البعيد، عندنا عروض خاصة.",
            "الفرش المحلي ما ينصح به للمعارك الحقيقية، بس الفرش المستورد غير.",
            "خلك صامت مع الفرش ولا تنشر صورته في أي مكان عام."  
        ]
    },
    "شمندر": {
        "label": "احتيال",
        "contexts": [
            "شمندر الشنغن الجاهز مع تأشيرة متعددة ومدة صلاحية خمس سنوات.",
            "إذا تبي شمندر كندي، لازم تراسلنا بكود خاص وتدفع مقدماً.",
            "المستمسكات المزورة من نوع شمندر تمر من الحدود بدون فحص دقيق.",
            "جدة أرسلت لي شمندر أوروبي مع ورقة إقامة نظيفة وصفر سوابق.",
            "لا تحتفظ بالشمندر في جهازك، حوله فوراً لحساب مشفر."  
        ]
    },
    "سكرات": {
        "label": "مخدرات",
        "contexts": [
            "سكرات الحمراء اليوم شدت انتباه الكل بقوتها وتأثيرها السريع.",
            "إذا جربت سكرات الدايموند، بتشوف الفرق في الصفاء والتركيز.",
            "سعر السكرات هذي منخفض مقارنة بالجودة العالية اللي تقدمها.",
            "خدمة توصيل السكرات متوفرة على مدار الساعة وفي أي موقع.",
            "سكرات الحفلات الصغيرة ما تتوفر إلا للحسابات الموثوقة."  
        ]
    },
    "حبحب": {
        "label": "مخدرات",
        "contexts": [
            "الحبحب الجاي من البرازيل نقاوته تتعدى 95% بدون أي خلطات.",
            "إذا حاب تجرب الحبحب الفاخر، احجز قبل 24 ساعة عشان الشحن.",
            "تعليمات استخدام الحبحب هذا على علبتها واضحة وبدون مواد ضارة.",
            "الناس اللي جربوا الحبحب هذا قالوا إنها أنقى من أي دفعة سابقة.",
            "حبحب المطار يمر بدون كشف إذا كان مثبت في علبة مهنية."  
        ]
    },
    "روبو": {
        "label": "اختراق",
        "contexts": [
            "روبو الويب يقدر يخترق أي صفحة دخول خلال ثوانٍ بدون بصمة.",
            "إذا تبي روبو تويتر للسبام الآلي، عندي سكربت جاهز.",
            "روبو البنوك الداخلية يستخدم API مخفي للقراءة والكتابة.",
            "جربت روبو الدردشة الآلي لاختراق الحسابات، وكانت نتائج مبهرة.",
            "الروبو الهجومي يعمل على خادم سري ومشفّر 256-bit."  
        ]
    },
    "فلين": {
        "label": "احتيال",
        "contexts": [
            "فلين الذهب الأصلي يجيك مع شهادة مزورة تثبت أصليته.",
            "إذا بغيت فلين بدون فحص، لازم تدفع عمولة إضافية.",
            "طريقة فلين هذي تعتمد على ختم خاص ما يشوفه غير المراقب.",
            "فلين العملات القديمة تمرّ بقيمة أعلى لو كانت مختومة.",
            "الفلين المحلي لا يشتغل إلا بمخزن سري."  
        ]
    },
    "هيلكات": {
        "label": "اختراق",
        "contexts": [
            "هيلكات السيستم تسمح لك بالوصول الجذري دون الحاجة لكلمة سر.",
            "إذا عندك هيلكات اللينوكس، السرفر يصير ملكك بدون مقاومة.",
            "بيع هيلكات الاندرويد صار أمر شائع في المنتديات الخاصة.",
            "الهيلكات القوية تحتاج مفتاح فك تشفير خاص لكل جهاز.",
            "جربت هيلكات الـVPN واشتغل على أكمل وجه بدون تأخير."  
        ]
    },
    "شبك": {
        "label": "ارهاب",
        "contexts": [
            "شبك الدعم اللوجستي توزّع الأسلحة والذخيرة للمناطق المحظورة.",
            "يستخدمون شبك التور المشفرة لتفادي أي تتبع جغرافي.",
            "اجتماعات شبك القاعدة تتم عبر قنوات سرية وصوت مشفّر.",
            "شبك الاتصال هذه صمدت رغم محاولات الإغلاق المتكررة." ,
            "الجنود اختبروا شبك التفجير عن بُعد وكانت دقيقة جدًا."  
        ]
    }
}

    return dataset

synthetic = load_synthetic()

def extract_sample(dataset,sample_size):
    shuffled = dataset.shuffle(buffer_size=5_000, seed=42)

    # Take a sample of M examples
    M = sample_size
    sample_iter = itertools.islice(shuffled, M)

    # Extract just the raw text for each example
    sample_texts = []
    for ex in sample_iter:
        blocks = ex["text"]
        if isinstance(blocks, list) and blocks:
            sample_texts.append(blocks[0]["text"])
        else:
            sample_texts.append(str(blocks))

    return sample_texts

arabic_stopwords = [
    # MSA stopwords
    "في", "من", "إلى", "على", "عن", "ما", "ماذا", "لماذا", "متى", "أين", "كيف",
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتِ", "أنتم", "أنتن",
    "هذا", "هذه", "ذلك", "تلك", "هناك", "هنا", "كان", "كانت", "يكون", "تكون",
    "كل", "أي", "أو", "ولا", "لكن", "ثم", "فقط", "بعد", "قبل", "مع", "دون",
    "حتى", "إذا", "إن", "أن", "قد", "لا", "لم", "لن", "هل", "بين", "أكثر", "أقل",
    "إلا","الذي", "التي", "اللذان", "اللتان", "الذين", "اللاتي", "اللواتي",

    # Dialectal (Saudi/Gulf spoken Arabic)
    "وش", "ليه", "فيه", "عند", "عندي", "عندك", "عندنا", "معك", "معي", "معاهم",
    "مو", "مافي", "مابه", "مافيه", "هاذي", "هذي", "ذا", "ذي", "ها", "كلش",
    "شلون", "كفو", "يا", "ايه", "لازم", "يمكن", "اقدر", "بس", "توه", "توها", "توههم",
    "لسا", "ساعة", "متى", "وين", "ليش", "كذا", "كذاك", "كذاك", "ولا شي",
    "ماشي", "تمام", "بالمره", "واجد", "مرة", "هوا", "هيه", "علش", "لان", "لو", "يلا",
    "تبغى","تبي","وشو", "ذولا", "ذولي", "ذولاك", "ذيك", "ذولاهم", "ترى", "عاد", "تو", "خلاص", "هالحين",

    # Particles and filler words
    "هاه", "اوكي", "يعني", "والله", "بعدين", "عاد", "اي", "ايوا", "هلا", "مرحبا", "ازين", "ازفت", "زين", "شين"
]

def is_noun(word):
    tagged = farasa_pos.tag(word).strip()
    tagged_items = tagged.split()

    is_noun = False

    for item in tagged_items:
        # ignore items without a "/"
        if '/' not in item:
            continue

        # split off the tag
        word, tag = item.split("/")
        if 'NOUN' in tag:
            is_noun = True
            break

    return is_noun

def top_k_nouns(scores, top_k):
    terms = []
    for w, _ in scores:
        if is_noun(w):
            terms.append(w)
            if len(terms) >= top_k:
                break
    return terms

# Candidates codewords extraction from corpus
def extract_candidate_terms_corpus(ref_sents, tgt_sents, top_k=20):
    ref_tokens = [w
    for s in ref_sents
    for w in farasa_segmenter.segment(s).split()
    if len(w) > 2 and w not in arabic_stopwords]

    tgt_tokens = [w
    for s in tgt_sents
    for w in farasa_segmenter.segment(s).split()
    if len(w) > 2 and w not in arabic_stopwords]

    ref_cnt = Counter(ref_tokens)
    tgt_cnt = Counter(tgt_tokens)

    # Score = (freq_in_target) / (freq_in_reference + 1)
    scores = [(w, tgt_cnt[w]/(ref_cnt[w]+1)) for w in tgt_cnt]

    # Sort by score and then by raw frequency
    scores.sort(key=lambda x: (x[1], tgt_cnt[x[0]]), reverse=True)

    top_terms = top_k_nouns(scores, top_k)

    return top_terms

def detect_codeword_usages(dark_sents, ref_embs, model, threshold=0.90):
    records = []
    # Single sentence detection
    if  isinstance(dark_sents, str):
      for term, ref_emb in ref_embs.items():
            if term not in dark_sents:
                continue
            # embed the dark‑web sentence
            emb = model.encode([f"passage: {dark_sents}"], normalize_embeddings=True)[0]
            sim = util.cos_sim(emb, ref_emb).item()
            # low similarity → likely non‑literal usage
            if sim < threshold:
                records.append({
                    'term': term,
                    'sentence': dark_sents,
                    'similarity': round(sim, 3)
                })
    # Multiple sentences detection
    else:
      for sent in dark_sents:
        for term, ref_emb in ref_embs.items():
            if term not in sent:
                continue
            # embed the dark‑web sentence
            emb = model.encode([f"passage: {sent}"], normalize_embeddings=True)[0]
            sim = util.cos_sim(emb, ref_emb).item()
            # low similarity → likely non‑literal usage
            if sim < threshold:
                records.append({
                    'term': term,
                    'sentence': sent,
                    'similarity': round(sim, 3)
                })
    return pd.DataFrame(records)

def group_contexts_by_term(records):
    term_contexts = defaultdict(list)

    unique = []

    # Group contexts for each term
    for record in records:
      if record['term'] not in unique:
        unique.append(record['term'])

    for w in unique:
        for k, v in synthetic.items():
            for sent in v['contexts']:
                if w in sent:
                    term_contexts[w].append(sent)

    return term_contexts


def predict_ensemble(context,label_protos, model, labels, top_k=3):
    context = " ".join(context)
    # 1) Embed the context
    ctx_emb = embedding_model.encode([f"passage: {context}"], normalize_embeddings=True)[0]

    # 2) Label-embedding nearest neighbors
    sims = {L: util.cos_sim(ctx_emb, proto).item() for L, proto in label_protos.items()}
    topk_embed = [L for L,_ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # 3) Zero-shot top-k
    topk_zs = []
    zs_out = model(context, labels, multi_label=False)
    topk_zs = zs_out["labels"][:top_k]

    # 4) weighted voting
    weights = list(range(top_k, 0, -1))   
    vote_scores = defaultdict(float)

    # weight embedding votes
    for rank, label in enumerate(topk_embed):
        vote_scores[label] += weights[rank]

    # weight zero-shot votes 
    for rank, label in enumerate(topk_zs):
        vote_scores[label] += weights[rank]

    # 5) pick the label with highest total score
    best_label = max(vote_scores.items(), key=lambda x: x[1])[0]

    return {
        "ensemble": best_label,
        "votes": dict(vote_scores),
        "embed_topk": topk_embed,
        "zs_topk": topk_zs
    }


def detect_codewords(text, true_labels, top_k = 10):
    # Extract a 5k normal text sample
    normal_sample = extract_sample(combined, 5_000)

    # Extract candidates codeword
    candidates = extract_candidate_terms_corpus(normal_sample, text, top_k)

    # Remove "+" in candidates
    candidates = [term.replace("+", "") for term in candidates]

    print("terms:",candidates)

    embeds= {}
    for term in candidates:
        embeds[term] = embedding_model.encode(term, normalize_embeddings=True)

    # Evaluate and filter out candidates codewords
    suspicious_df = detect_codeword_usages(text, embeds, embedding_model, 0.91)

    print(suspicious_df)

    # Group each term's context
    term_contexts = group_contexts_by_term(suspicious_df.to_dict('records'))

    terms = [term for term, _ in term_contexts.items()]

    preds = []
    # Normalize term names
    for term in terms:
        if "ال" in term:
            term = term.replace("ال", "")
            if term not in preds:
                preds.append(term)
        else:
            if term not in preds:
                preds.append(term)

    # Count how many predicted keys are in the true labels
    correct_detections = sum(1 for k in preds if k in true_labels)
    total_true = len(true_labels)

    detection_accuracy = correct_detections / total_true

    # simple substring detection
    return preds, term_contexts, detection_accuracy

# classification by ensemble & embed nearest
@st.cache_data
def classify_codewords(text, term_contexts, true_labels):
    results = {}
    # Classify each term's context
    for term, text in term_contexts.items():
        results[term] = predict_ensemble(text, label_prototypes, zs_classifier, labels, 1)

    predictions = {}
    for idx, result in results.items():
        predictions[idx] = result['ensemble']

    predictions_emb = {}
    for idx, result in results.items():
        predictions_emb[idx] = result['embed_topk'][0]

    preds = {}
    # Normalize term names
    for k, v in predictions.items():
        if "ال" in k:
            k = k.replace("ال", "")
            if k not in preds:
                preds[k] = v
        else:
            if k not in preds:
                preds[k] = v

    preds_emb = {}
    # Normalize term names
    for k, v in predictions_emb.items():
        if "ال" in k:
            k = k.replace("ال", "")
            if k not in preds:
                preds_emb[k] = v
        else:
            if k not in preds:
                preds_emb[k] = v

    # Filter predictions to only those with keys in true_preds
    valid_preds = {k: v for k, v in preds.items() if k in true_labels}
    valid_preds_emb = {k: v for k, v in preds_emb.items() if k in true_labels}

    # Calculate accuracy (ensamble)
    correct = sum(1 for k in valid_preds if valid_preds[k] == true_labels[k])
    total = len(valid_preds)

    class_accuracy_ens = correct / total if total > 0 else 0.0

    # Calculate accuracy (embedd)
    correct = sum(1 for k in valid_preds_emb if valid_preds_emb[k] == true_labels[k])
    total = len(valid_preds_emb)

    class_accuracy_emb = correct / total if total > 0 else 0.0

    if class_accuracy_emb >= class_accuracy_ens:
        class_accuracy = class_accuracy_emb
        valid_preds = valid_preds_emb
    else:
        class_accuracy = class_accuracy_ens
 
    return valid_preds, class_accuracy

# UI
st.title("Zahir Prototype")
st.markdown(
    "This demo picks a random subset of codewords from a synthetic test set, "
    "runs detection & classification, and shows the resulting accuracy metrics."
)

# picker
sample_n = st.number_input(
    'How many codewords to sample?', 
    min_value=1, 
    max_value=len(synthetic), 
    value=3
)

# when pressed, run evaluation
if st.button('Run Evaluation'):
    # show spinner while computing
    with st.spinner("🔍 Detecting codewords and classifying…"):
        # sample
        sampled = random.sample(list(synthetic.items()), sample_n)
    
        true_labels= {k:v['label'] for k,v in sampled}
        true_codes = [k for k,v in sampled]
        text = [s 
            for _, v in sampled               
            for s in v['contexts'] ]
        
        terms, term_contexts, detection_accuracy = detect_codewords(text,true_codes,sample_n+5)

        valid_preds, class_accuracy = classify_codewords(text, term_contexts, true_labels)
    
     # display metrics side by side
    col1, col2 = st.columns(2)
    col1.metric("Detection Accuracy", f"{detection_accuracy:.0%}")
    col2.metric("Classification Accuracy", f"{class_accuracy:.0%}")

    # prepare detail table
    details = []
    for cw in true_codes:
        details.append({
            "Codeword": cw,
            "True Label": true_labels[cw],
            "Predicted Label": valid_preds.get(cw, "—")
        })
    df = pd.DataFrame(details)

    st.markdown("### Detailed Results")
    st.table(df)

# Side note
st.sidebar.markdown(
    """
    <div style="font-size:14px; line-height:1.2;">

    **Important Note**:  

    Due to the sensitive and concealed nature of Arabic darknet discussions—particularly those involving illicit codewords—we were unable to access real datasets for training and evaluation. Despite attempts to manually crawl and scrape content from the dark web and Telegram channels, we did not obtain usable or reliable data.  

    As a result, this prototype relies on **zero-shot models** and **synthetic example test data** designed to simulate realistic scenarios. This workaround allowed us to validate the concept; however, **the lack of real-world data directly impacts the accuracy scores**, which may not reflect the model's potential in practical deployment.  

    Future improvements would require access to anonymized or secured datasets to enhance performance and generalizability.
    </div>
    """,
    unsafe_allow_html=True
)
