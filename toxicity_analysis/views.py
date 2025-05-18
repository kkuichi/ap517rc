import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
from .models import UserRating
from django.shortcuts import render, redirect

# Načítanie modelu
model_name = "minuva/MiniLMv2-toxic-jigsaw"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# Funkcia na predpovedanie pravdepodobnosti
def predict_proba(texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
        probabilities = torch.sigmoid(outputs).numpy()
    return probabilities

def generate_integrated_gradients_explanation(text):
    try:
        # Tokenizácia textu
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Získajte vložené hodnoty z modelu
        embeddings = model.get_input_embeddings()(input_ids)

        # Vytvorenie funkcie pre Captum
        def forward_func(input_embeds):
            return model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits

        # Vytvorenie vysvetľujúcej funkcie pre Integrated Gradients
        ig = IntegratedGradients(forward_func)

        # Výpočet atribútov
        attributions, _ = ig.attribute(embeddings, target=0, return_convergence_delta=True)
        attributions = attributions.sum(dim=-1).squeeze(0)  # Sumarizácia na poslednej osi
        attributions = attributions / torch.norm(attributions)  # Normalizujte hodnoty

        # Prevod tokenov a atribútov
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Odstránenie servisných tokenov ([CLS], [SEP], [PAD])
        filtered_tokens = []
        filtered_attributions = []
        for token, attr in zip(tokens, attributions.tolist()):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Vynechajte servisné tokeny
            filtered_tokens.append(token)
            filtered_attributions.append(attr)

        # Spojte čiastkové tokeny do slov
        merged_tokens, merged_attributions = merge_subtokens(filtered_tokens, filtered_attributions)

        # Vytvorenie grafu dôležitosti slov
        plt.figure(figsize=(10, 6))
        bars = plt.barh(merged_tokens, merged_attributions, color="skyblue", edgecolor='black')

        # Pridanie číselných hodnôt nad stĺpce
        for bar, attr in zip(bars, merged_attributions):
            plt.text(
                bar.get_width() + 0.01,  # Odsunutie textu
                bar.get_y() + bar.get_height() / 2,  # Vycentrovanie textu na výšku
                f"{attr:.2f}",  # Formátovanie hodnoty na dve desatinné miesta
                va='center',  # Vertikálne zarovnanie
                ha='left',  # Vodorovné zarovnanie
                fontsize=9  # Veľkosť písma
            )

        # Pridanie nadpisov záhlavia a osí
        plt.title("Integrated Gradients: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Attribution Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        # Odstránenie mriežky a prispôsobenie vzhľadu
        plt.grid(False)
        plt.tight_layout()

        # Uloženie grafu ako base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating Integrated Gradients explanation: {str(e)}")
        return None



def generate_lime_explanation(text):
    try:
        # Generovanie vysvetlivky LIME
        explainer = LimeTextExplainer(class_names=labels)
        # Generovanie vysvetlenia
        explanation = explainer.explain_instance(
            text,
            lambda x: predict_proba(x),
            num_features=10,
            labels=[0]  # Analýza len prvej triedy (toxicita)
        )
        # Výber dôležitosti slov
        lime_features = explanation.as_list(label=0)  # Získanie zoznamu slov a ich dôležitosti
        tokens, importances = zip(*lime_features)  # Rozdelenie na tokeny a hodnoty
        # Vytvorenie grafu
        plt.figure(figsize=(10, 6))
        bars = plt.barh(tokens, importances, color="skyblue", edgecolor='black')  # Jedna farba pre všetky slová
        # Pridanie číselných hodnôt nad stĺpce
        for bar, imp in zip(bars, importances):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.2f}",
                va='center',
                ha='left',
                fontsize=9
            )
        # Pridanie nadpisu a nadpisov osí
        plt.title("LIME: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)
        # Odstránenie mriežky a prispôsobenie vzhľadu
        plt.grid(False)  # Odstrániť mriežku
        plt.tight_layout()  # Automatické prispôsobenie prvkov
        # Uloženie grafu ako base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')  # Odstránenie nadmerného odsadenia
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic
    except Exception as e:
        print(f"Error generating LIME explanation: {str(e)}")
        return None



# Generovanie vysvetlenia SHAP
def generate_shap_explanation(text):
    try:
        explainer = shap.Explainer(lambda x: predict_proba(x), shap.maskers.Text(tokenizer))
        shap_values = explainer([text])
        encoded = tokenizer(text, return_offsets_mapping=True)
        words = tokenizer.convert_ids_to_tokens(encoded["input_ids"])[1:-1]  # Odstránenie [CLS] a [SEP]
        offsets = encoded["offset_mapping"][1:-1]
        importances = shap_values.values[0].sum(axis=1)  # Sumarizácia hodnôt na osiach

        # Spojte čiastkové toky do slov
        word_importance = {}
        current_word = ""
        current_importance = 0
        previous_end = -1
        for (token, importance, (start, end)) in zip(words, importances, offsets):
            if start != previous_end and current_word:
                word_importance[current_word] = current_importance
                current_word = ""
                current_importance = 0
            current_word += token.replace("##", "")
            current_importance += importance
            previous_end = end
        if current_word:
            word_importance[current_word] = current_importance

        # Prevod na DataFrame
        df = pd.DataFrame(list(word_importance.items()), columns=["Word", "Importance"])

        # Vytvorenie grafu
        plt.figure(figsize=(10, 6))
        bars = plt.barh(df["Word"], df["Importance"], color="skyblue", edgecolor='black')

        # Pridanie číselných hodnôt nad stĺpce
        for bar, imp in zip(bars, df["Importance"]):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.2f}",
                va='center',
                ha='left',
                fontsize=9
            )

        # Pridanie nadpisu a nadpisov osí
        plt.title("SHAP: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        plt.grid(False)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating SHAP explanation: {str(e)}")
        return None




# Generovanie vysvetlení DeepLift
def generate_deeplift_explanation(text):
    try:
        # Tokenizácia textu
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Získanie vkladov z modelu
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(input_ids)

        # Vytvorenie obalovej triedy pre forward_func
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, embeddings, attention_mask=None):
                outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                return outputs.logits

        wrapped_model = ModelWrapper(model)

        # Vytvorenie vysvetľovacej triedy DeepLift
        deep_lift = DeepLift(wrapped_model)

        # Definovanie referenčných hodnôt
        baseline_embeddings = torch.zeros_like(embeddings)

        # Výpočet atribútov pomocou DeepLIFT
        attributions = deep_lift.attribute(
            inputs=embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask,),
            target=0
        )

        # Prevedenie atribútov do čitateľného formátu
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        # Vizualizácia výsledku
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Odstránenie servisných tokenov ([CLS], [SEP], [PAD])
        filtered_tokens = []
        filtered_attributions = []
        for token, attr in zip(tokens, attributions.tolist()):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            filtered_tokens.append(token)
            filtered_attributions.append(attr)

        # Spojenie čiastkových tokenov do slov
        merged_tokens, merged_attributions = merge_subtokens(filtered_tokens, filtered_attributions)

        # Vytvorenie grafu dôležitosti slov
        plt.figure(figsize=(10, 6))
        bars = plt.barh(merged_tokens, merged_attributions, color="skyblue", edgecolor='black')

        for bar, attr in zip(bars, merged_attributions):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{attr:.2f}",
                va='center',
                ha='left',
                fontsize=9
            )

        plt.title("DeepLift: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Attribution Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        plt.grid(False)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating DeepLift explanation: {str(e)}")
        return None



# Funkcia na spájanie čiastkových tokenov do slov
def merge_subtokens(tokens, attributions):
    merged_tokens = []
    merged_attributions = []
    current_token = ""
    current_attr = 0.0

    for token, attr in zip(tokens, attributions):
        if token.startswith("##"):
            current_token += token[2:]
            current_attr += attr
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_attributions.append(current_attr)
            current_token = token
            current_attr = attr

    if current_token:
        merged_tokens.append(current_token)
        merged_attributions.append(current_attr)

    return merged_tokens, merged_attributions

def analyze_toxicity(request):
    lime_explanation_image = None
    shap_explanation_image = None
    deeplift_explanation_image = None
    integrated_gradients_explanation_image = None
    model_verdict = None
    email = ""
    text = ""

    if request.method == "POST":
        text = request.POST.get("text", "")
        email = request.POST.get("email", "").strip()  # Získanie e-mailu

        if text:
            probabilities = predict_proba([text])[0]
            toxic_probability = probabilities[0]  # Pravdepodobnosť toxicity
            model_verdict = "Toxic" if toxic_probability >= 0.5 else "Non-toxic"

            # Generovanie vysvetlení
            try:
                lime_explanation_image = generate_lime_explanation(text)
            except Exception as e:
                lime_explanation_image = f"<p>Error when generating an explanation of LIME: {str(e)}</p>"
            try:
                shap_explanation_image = generate_shap_explanation(text)
            except Exception as e:
                shap_explanation_image = f"<p>Error when generating a SHAP explanation: {str(e)}</p>"
            try:
                deeplift_explanation_image = generate_deeplift_explanation(text)
            except Exception as e:
                deeplift_explanation_image = f"<p>Error while generating an explanation of DeepLift: {str(e)}</p>"
            try:
                integrated_gradients_explanation_image = generate_integrated_gradients_explanation(text)
            except Exception as e:
                integrated_gradients_explanation_image = f"<p>Error when generating an explanation of Integrated Gradients: {str(e)}</p>"

        # Kontrola, či boli odoslané výsledky
        if "lime_explanation_goodness" in request.POST:
            # Hodnoty pre LIME
            lime_explanation_goodness = int(request.POST.get("lime_explanation_goodness", 1))
            lime_user_satisfaction = int(request.POST.get("lime_user_satisfaction", 1))
            lime_user_understanding = int(request.POST.get("lime_user_understanding", 1))
            lime_user_curiosity = int(request.POST.get("lime_user_curiosity", 1))
            lime_user_trust = int(request.POST.get("lime_user_trust", 1))
            lime_system_controllability = int(request.POST.get("lime_system_controllability", 1))
            lime_user_productivity = int(request.POST.get("lime_user_productivity", 1))

            # Hodnoty pre SHAP
            shap_explanation_goodness = int(request.POST.get("shap_explanation_goodness", 1))
            shap_user_satisfaction = int(request.POST.get("shap_user_satisfaction", 1))
            shap_user_understanding = int(request.POST.get("shap_user_understanding", 1))
            shap_user_curiosity = int(request.POST.get("shap_user_curiosity", 1))
            shap_user_trust = int(request.POST.get("shap_user_trust", 1))
            shap_system_controllability = int(request.POST.get("shap_system_controllability", 1))
            shap_user_productivity = int(request.POST.get("shap_user_productivity", 1))

            # Hodnoty pre DeepLift
            deeplift_explanation_goodness = int(request.POST.get("deeplift_explanation_goodness", 1))
            deeplift_user_satisfaction = int(request.POST.get("deeplift_user_satisfaction", 1))
            deeplift_user_understanding = int(request.POST.get("deeplift_user_understanding", 1))
            deeplift_user_curiosity = int(request.POST.get("deeplift_user_curiosity", 1))
            deeplift_user_trust = int(request.POST.get("deeplift_user_trust", 1))
            deeplift_system_controllability = int(request.POST.get("deeplift_system_controllability", 1))
            deeplift_user_productivity = int(request.POST.get("deeplift_user_productivity", 1))

            # Hodnoty pre Integrated Gradients
            integrated_gradients_explanation_goodness = int(request.POST.get("integrated_gradients_explanation_goodness", 1))
            integrated_gradients_user_satisfaction = int(request.POST.get("integrated_gradients_user_satisfaction", 1))
            integrated_gradients_user_understanding = int(request.POST.get("integrated_gradients_user_understanding", 1))
            integrated_gradients_user_curiosity = int(request.POST.get("integrated_gradients_user_curiosity", 1))
            integrated_gradients_user_trust = int(request.POST.get("integrated_gradients_user_trust", 1))
            integrated_gradients_system_controllability = int(request.POST.get("integrated_gradients_system_controllability", 1))
            integrated_gradients_user_productivity = int(request.POST.get("integrated_gradients_user_productivity", 1))

            # Uloženie výsledkov do databázy
            UserRating.objects.create(
                email=email or None,
                text=text,
                lime_explanation_goodness=lime_explanation_goodness,
                lime_user_satisfaction=lime_user_satisfaction,
                lime_user_understanding=lime_user_understanding,
                lime_user_curiosity=lime_user_curiosity,
                lime_user_trust=lime_user_trust,
                lime_system_controllability=lime_system_controllability,
                lime_user_productivity=lime_user_productivity,
                shap_explanation_goodness=shap_explanation_goodness,
                shap_user_satisfaction=shap_user_satisfaction,
                shap_user_understanding=shap_user_understanding,
                shap_user_curiosity=shap_user_curiosity,
                shap_user_trust=shap_user_trust,
                shap_system_controllability=shap_system_controllability,
                shap_user_productivity=shap_user_productivity,
                deeplift_explanation_goodness=deeplift_explanation_goodness,
                deeplift_user_satisfaction=deeplift_user_satisfaction,
                deeplift_user_understanding=deeplift_user_understanding,
                deeplift_user_curiosity=deeplift_user_curiosity,
                deeplift_user_trust=deeplift_user_trust,
                deeplift_system_controllability=deeplift_system_controllability,
                deeplift_user_productivity=deeplift_user_productivity,
                integrated_gradients_explanation_goodness=integrated_gradients_explanation_goodness,
                integrated_gradients_user_satisfaction=integrated_gradients_user_satisfaction,
                integrated_gradients_user_understanding=integrated_gradients_user_understanding,
                integrated_gradients_user_curiosity=integrated_gradients_user_curiosity,
                integrated_gradients_user_trust=integrated_gradients_user_trust,
                integrated_gradients_system_controllability=integrated_gradients_system_controllability,
                integrated_gradients_user_productivity=integrated_gradients_user_productivity,
            )

            #  Presmerovanie používateľa na tú istú stránku na obnovenie formulára
            return redirect("analyze_toxicity")

    return render(request, "toxicity/index.html", {
        "lime_explanation_image": lime_explanation_image,
        "shap_explanation_image": shap_explanation_image,
        "deeplift_explanation_image": deeplift_explanation_image,
        "integrated_gradients_explanation_image": integrated_gradients_explanation_image,
        "model_verdict": model_verdict,
        "email": email,
        "text": text
    })