import pandas as pd
import numpy as np
import joblib
import os
import warnings
import time
warnings.filterwarnings("ignore")

class AyurPredictSystem:
    def __init__(self, model_path, encoder_path, data_path):
        """Initialize AyurPredict system with model, encoders, and data"""
        print("🌿 Initializing AyurPredict System...")

        start = time.time()

        # Load ML model and metadata
        self.model_artifacts = joblib.load(model_path)
        self.model = self.model_artifacts["model"]
        self.feature_columns = self.model_artifacts["feature_columns"]

        # Load label encoders and main dataset
        self.encoders = joblib.load(encoder_path)
        self.data = pd.read_csv(data_path)

        # Build reverse mappings (for decoding)
        self.reverse_mappings = {}
        for col, enc in self.encoders.items():
            if hasattr(enc, "classes_"):
                self.reverse_mappings[col.replace("_encoded", "")] = {
                    i: cls for i, cls in enumerate(enc.classes_)
                }

        # Prepare helper mappings and static knowledge
        self._define_properties()
        self._define_symptom_mapping()
        self._define_risks()

        print(f"✅ AyurPredict system loaded successfully in {round(time.time()-start,2)}s!")
        print(f"   🌿 Herbs available: {len(self.get_herbs())}")
        print(f"   🎯 Targets available: {len(self.get_targets())}")

    # --------------------------------------------------------------------
    # Basic Utility Functions
    # --------------------------------------------------------------------
    def get_herbs(self):
        return list(self.reverse_mappings.get("Herb_Name", {}).values())

    def get_targets(self):
        return list(self.reverse_mappings.get("Target_Name", {}).values())

    def encode(self, name, key):
        """Encode herb or target name into numerical code"""
        mapping = self.reverse_mappings.get(key, {})
        for code, label in mapping.items():
            if label.lower() == name.lower():
                return code
        return None

    def decode(self, code, key):
        """Decode numeric code back to herb/target name"""
        mapping = self.reverse_mappings.get(key, {})
        return mapping.get(code, f"Unknown {key} ({code})")

    # --------------------------------------------------------------------
    # Static Knowledge
    # --------------------------------------------------------------------
    def _define_properties(self):
        self.property_columns = [col for col in self.data.columns if "Property_" in col]
        self.property_descriptions = {
            "Property_neuroprotective": "Protects nerve cells from damage",
            "Property_antioxidant": "Fights oxidative stress",
            "Property_adaptogenic": "Helps the body resist stress",
            "Property_anti-inflammatory": "Reduces inflammation",
            "Property_antimicrobial": "Fights infections",
            "Property_hepatoprotective": "Protects liver function",
            "Property_digestive": "Supports digestion",
            "Property_cardioprotective": "Supports heart health",
            "Property_hypoglycemic": "Lowers blood sugar",
            "Property_hypolipidemic": "Lowers cholesterol",
            "Property_nootropic": "Improves memory and focus",
        }

    def _define_symptom_mapping(self):
        """Maps symptoms to key therapeutic properties"""
        self.symptom_to_properties = {
            'anxiety': ['Property_anxiolytic', 'Property_adaptogenic', 'Property_antidepressant'],
            'stress': ['Property_adaptogenic', 'Property_antioxidant'],
            'insomnia': ['Property_sedative', 'Property_hypnotic', 'Property_calming'],
            'depression': ['Property_antidepressant', 'Property_mood_stabilizer'],
            'memory loss': ['Property_nootropic', 'Property_neuroprotective'],
            'fatigue': ['Property_metabolism_booster', 'Property_stimulant'],
            'cold': ['Property_antiviral', 'Property_immunomodulatory', 'Property_expectorant'],
            'fever': ['Property_antipyretic', 'Property_anti_inflammatory'],
            'infection': ['Property_antimicrobial', 'Property_antiviral', 'Property_antifungal'],
            'pain': ['Property_analgesic', 'Property_anti_inflammatory'],
            'asthma': ['Property_bronchodilator', 'Property_anti_inflammatory'],
            'allergy': ['Property_antiallergic', 'Property_anti_inflammatory'],
        }

    def _define_risks(self):
        """Known herb safety data"""
        self.safety_info = {
            "Ashwagandha": "Avoid during pregnancy. May cause drowsiness.",
            "Tulsi": "Generally safe. May thin blood.",
            "Brahmi": "High doses can cause nausea.",
            "Giloy": "Avoid in autoimmune disorders.",
            "Shankhpushpi": "Mild sedative; avoid with alcohol.",
            "Neem": "Avoid during pregnancy; may lower blood sugar.",
            "Amla": "Generally safe; may cause mild acidity if taken empty stomach."
        }

    # --------------------------------------------------------------------
    # Core ML-based Prediction Helpers
    # --------------------------------------------------------------------
    def _create_features(self, herb_code, target_code):
        """Construct feature vector dynamically for model prediction"""
        features = {
            "Herb_Name_encoded": herb_code,
            "Target_Name_encoded": target_code,
            "Compound_Name_encoded": 0,
            "Action_Type_encoded": 1,
            "Species_encoded": 1,
            "Data_Quality_Score": 0.85,
            "Confidence_Score": 0.75
        }

        # Add zero baseline for all property columns
        for col in self.property_columns:
            features[col] = 0

        # Interaction features
        features["Herb_Target_Interaction"] = herb_code * target_code
        features["Herb_Target_Specificity"] = herb_code / (target_code + 1)

        # Convert to DataFrame aligned to model
        df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_columns]

    def _predict_bioactivity(self, herb_code, target_codes):
        """Batch predict bioactivity scores (optimized)"""
        feature_list = []
        for tgt_code in target_codes:
            feature_list.append(self._create_features(herb_code, tgt_code).iloc[0])
        X = pd.DataFrame(feature_list)
        preds = self.model.predict(X)
        return np.mean(preds)

    # --------------------------------------------------------------------
    # Pathway 1 — Researcher Mode (Herb → Targets)
    # --------------------------------------------------------------------
    def researcher_pathway(self, herb_name):
        """Predict targets, compounds, effects for given herb"""
        print(f"\n🔬 RESEARCHER MODE: '{herb_name}' → Targets")

        herb_code = self.encode(herb_name, "Herb_Name")
        if herb_code is None:
            print(f"❌ Herb '{herb_name}' not found in dataset.")
            return None

        herb_data = self.data[self.data["Herb_Name_encoded"] == herb_code]

        # Extract known compounds/pathways
        compounds = herb_data["Compound_Name_encoded"].unique().tolist()
        pathways = herb_data["Pathway"].unique().tolist() if "Pathway" in herb_data.columns else []

        # Predict target interactions (limit to top 50 for speed)
        targets = list(self.reverse_mappings.get("Target_Name", {}).items())[:50]
        predictions = [(tgt_name, self._predict_bioactivity(herb_code, [tgt_code]))
                       for tgt_code, tgt_name in targets]

        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

        properties = {
            prop.replace("Property_", "").title(): "✅"
            for prop in self.property_columns
            if prop in herb_data.columns and herb_data[prop].sum() > 0
        }

        return {
            "herb": herb_name,
            "predicted_targets": predictions,
            "compounds": compounds[:5],
            "pathways": pathways[:5],
            "therapeutic_effects": list(properties.keys()),
            "safety": self.safety_info.get(herb_name, "No major side effects known.")
        }

    # --------------------------------------------------------------------
    # Pathway 2 — Symptom Mode (Symptom → Herb Recommendation)
    # --------------------------------------------------------------------
    def symptom_pathway(self, symptom):
        """Given a symptom, find best herb candidates"""
        print(f"\n💊 SYMPTOM MODE: '{symptom}'")

        properties = self.symptom_to_properties.get(symptom.lower(), [])
        if not properties:
            print(f"❌ No related properties found for: {symptom}")
            return None

        matched_herbs = []
        targets = list(range(min(30, len(self.get_targets()))))  # limit to 30 targets

        for herb_name in self.get_herbs():
            herb_code = self.encode(herb_name, "Herb_Name")
            if herb_code is None:
                continue
            herb_data = self.data[self.data["Herb_Name_encoded"] == herb_code]

            # Quick skip if no relevant property
            if not any(prop in herb_data.columns and herb_data[prop].sum() > 0 for prop in properties):
                continue

            # Batch predict mean bioactivity for selected targets
            score = self._predict_bioactivity(herb_code, targets)
            matched_herbs.append((herb_name, score))

        # Rank herbs by model-predicted interaction score
        matched_herbs = sorted(matched_herbs, key=lambda x: x[1], reverse=True)[:10]

        return {
            "symptom": symptom,
            "related_properties": [p.replace("Property_", "").title() for p in properties],
            "recommended_herbs": [
                {"herb": h, "predicted_strength": round(s, 3),
                 "risk": self.safety_info.get(h, "No major side effects known.")}
                for h, s in matched_herbs
            ]
        }

# --------------------------------------------------------------------
# MAIN EXECUTION (for testing standalone)
# --------------------------------------------------------------------
def main():
    BASE = r"c:\Users\HP\Downloads\minor_project_frontend\minor_project_frontend\model"
    MODEL_PATH = os.path.join(BASE, "ayurpredict_trustworthy_optimized_model.pkl")
    ENCODER_PATH = os.path.join(BASE, "feature_encoders.pkl")
    DATA_PATH = os.path.join(BASE, "ayurpredict_model_ready.csv")

    if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, DATA_PATH]):
        print("❌ Missing model or data files.")
        return

    system = AyurPredictSystem(MODEL_PATH, ENCODER_PATH, DATA_PATH)

    print("\n" + "="*80)
    print("🌿 AYURPREDICT - AI HERB & SYMPTOM PREDICTION SYSTEM")
    print("="*80)

    while True:
        print("\nChoose a pathway:")
        print("1️⃣ Researcher Pathway (Herb → Targets)")
        print("2️⃣ Symptom Pathway (Symptom → Herbs)")
        print("3️⃣ Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            herb = input("\nEnter herb name: ").strip()
            result = system.researcher_pathway(herb)
            if result:
                print(f"\n🌿 Herb: {result['herb']}")
                print(f"🎯 Predicted Targets: {[t[0] for t in result['predicted_targets']]}")
                print(f"💊 Compounds: {result['compounds']}")
                print(f"🧬 Pathways: {result['pathways']}")
                print(f"🩺 Therapeutic Effects: {result['therapeutic_effects']}")
                print(f"⚠️ Safety Info: {result['safety']}")

        elif choice == "2":
            symptom = input("\nEnter symptom: ").strip()
            result = system.symptom_pathway(symptom)
            if result:
                print(f"\n💊 Symptom: {result['symptom']}")
                print(f"🔬 Related Properties: {result['related_properties']}")
                print("🌿 Recommended Herbs:")
                for i, herb_info in enumerate(result["recommended_herbs"], 1):
                    print(f"   {i}. {herb_info['herb']} (Predicted Strength: {herb_info['predicted_strength']})")
                    print(f"      ⚠️ Risk: {herb_info['risk']}")

        elif choice == "3":
            print("👋 Exiting AyurPredict. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
