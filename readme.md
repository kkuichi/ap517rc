# Cieľ práce

Cieľom tejto štúdie bolo preskúmať a porovnať metódy vysvetľovania modelov machine learning (LIME, SHAP, DeepLift, Integrated Gradients) na zlepšenie transparentnosti a dôveryhodnosti riešenia modelu klasifikácie toxicity textu.  Projekt umožňuje používateľom:
1. Vložiť text a získať verdikt modelu o jeho toxicite.
2. Vizualizovať dôležitosť slov v texte pomocou rôznych metód vysvetľovania (LIME, SHAP, DeepLift, Integrated Gradients).
3. Vyhodnotiť kvalitu vysvetlenia podľa niekoľkých kritérií.
4. Uložiť hodnotenia používateľov do databázy na ďalšie analýzy.

---

# Štruktúra projektu

## 1. **Popis funkčnosti**
Projekt sa skladá z dvoch hlavných častí:
- **Model a metódy vysvetľovania**: Na klasifikáciu textov na toxické a netoxické sa používa vopred natrénovaný model `MiniLMv2-toxic-jigsaw'. Na vysvetlenie riešení modelu sa používajú tieto metódy:
  - **LIME**: Lokálna interpretácia modelov.
  - **SHAP**: Metóda založená na teórii kooperatívnych hier.
  - **DeepLift**: Metóda atribúcie, ktorá počíta vplyv vstupných prvkov na výstup modelu.
  - **Integrované gradienty**: Metóda, ktorá analyzuje gradienty modelu.

- **Webové rozhranie**: Implementované pomocou rámca Django. Rozhranie umožňuje používateľom zadávať text, získavať vysvetlenia a hodnotiť ich kvalitu.

---

## 2. **Hlavné súbory**

### `views.py`.
Súbor `views.py` obsahuje logiku na spracovanie požiadaviek od používateľa. Hlavnou funkciou je `analyse_toxicity`, ktorá:
1. Prijíma text a e-mail od používateľa.
2. Vygeneruje modelový verdikt o toxicite textu.
3. Vytvára vizualizácie vysvetlení pomocou metód LIME, SHAP, DeepLift a Integrated Gradients.
4. Ukladá hodnotenia používateľov do databázy.

### `index.html`.
Súbor `index.html` je zodpovedný za zobrazenie webového rozhrania. Hlavnými prvkami sú:
1. **Formulár na zadávanie textu**:
   - Pole na zadávanie textu (povinné).
   - Pole na zadanie e-mailu (nepovinné).
2. **Výsledky analýzy**:
   - Verdikt modelu (toxický/netoxický).
   - Vysvetľujúce vizualizácie pre každú metódu (LIME, SHAP, DeepLift, Integrated Gradients).
3. **Škály hodnotenia**:
   - Používateľ môže hodnotiť kvalitu vysvetlení podľa niekoľkých kritérií (napr. zrozumiteľnosť, spokojnosť, dôvera v model).

---

## 3. **Jupyter Notebook**

Priečinok `notebook` obsahuje dva súbory Jupyter Notebook:
- **`bp.ipynb`**: Obsahuje implementácie funkcií pre vysvetľovacie metódy (LIME, SHAP, DeepLift, Integrated Gradients) a metriku vernosti (Faithfulness). Tento súbor sa používa na testovanie a ladenie vysvetľovacích metód.
- **`results.ipynb`**: Obsahuje výsledky porovnania vysvetlení po subjektívnych hodnoteniach používateľov. Spracúva sa tu aj dátový rámec `output.csv`, ktorý obsahuje údaje o výsledkoch analýz.

---

## 4. **Technológie**.

- **Python**: Hlavný programovací jazyk.
- **Django**: Framework na vytváranie webových aplikácií.
- **Transformers (Hugging Face)**: Knižnica na prácu s predtrénovaným modelom.
- **LIME, SHAP, Captum**: Knižnice na vysvetlenie modelu.
- **Matplotlib**: Knižnica na vizualizáciu údajov.
- **SQLite**: Databáza na ukladanie hodnotení používateľov.

---

## 5.  **Začiatok projektu**

1.  Nastavte závislosti:
   ```bash
   pip install -r requirements.txt
   ```

2. Aplikujte migrácie:
   ```bash
   python manage.py migrate
   ```

3. Spustite server:
   ```bash
   python manage.py runserver
   ```

4. Otvorte prehliadač a prejdite na:
   ```
   http://127.0.0.1:8000/
   ```

---

## 6. **Funkcie webového rozhrania**.

- Používateľ môže zadať text a získať verdikt modelu o jeho toxicite.
- Pre každú metódu vysvetlenia (LIME, SHAP, DeepLift, Integrated Gradients) je k dispozícii vizualizácia dôležitosti slov.
- Po analýze môže používateľ zhodnotiť kvalitu vysvetlení podľa niekoľkých kritérií. Hodnotenia sa ukladajú do databázy.

--- 


