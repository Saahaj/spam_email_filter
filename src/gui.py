import sys
import os
import tkinter as tk
from tkinter import messagebox
import joblib
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'spam_detector_model.pkl'
VECTORIZER_PATH = PROJECT_ROOT / 'tfidf_vectorizer.pkl'

class SpamFilterAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Filter AI — Inspector")
        self.root.geometry("640x600")
        self.root.resizable(False, False)
        
        # Color Palette
        self.BG_COLOR = '#F8FAFC'      # Slate 50 background
        self.CARD_BG = '#FFFFFF'       # White card base
        self.PRIMARY = '#4F46E5'       # Indigo accent
        self.PRIMARY_HOVER = '#4338CA'
        self.TEXT_MAIN = '#1E293B'     # Slate 800
        self.TEXT_MUTED = '#64748B'    # Slate 500
        
        self.root.configure(bg=self.BG_COLOR)
        self._build_ui()

        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _build_ui(self):
        # Header Banner
        header = tk.Frame(self.root, bg=self.PRIMARY, pady=18)
        header.pack(fill='x')
        
        title = tk.Label(
            header, 
            text="🛡️ Spam Filter AI", 
            font=("Segoe UI", 20, "bold"), 
            fg='#FFFFFF', 
            bg=self.PRIMARY
        )
        title.pack()
        
        subtitle = tk.Label(
            header, 
            text="Instant machine learning email analysis", 
            font=("Segoe UI", 10), 
            fg='#E0E7FF', 
            bg=self.PRIMARY
        )
        subtitle.pack(pady=(2, 0))

        # Main Container
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR, padx=25, pady=20)
        main_frame.pack(fill='both', expand=True)

        # Content Card
        card = tk.Frame(
            main_frame, 
            bg=self.CARD_BG, 
            relief='flat', 
            highlightbackground='#E2E8F0', 
            highlightthickness=1
        )
        card.pack(fill='both', expand=True)

        card_inner = tk.Frame(card, bg=self.CARD_BG, padx=20, pady=20)
        card_inner.pack(fill='both', expand=True)

        instructions = tk.Label(
            card_inner, 
            text="Paste email text to analyze:", 
            font=("Segoe UI", 11, "bold"), 
            bg=self.CARD_BG, 
            fg=self.TEXT_MAIN,
            anchor='w'
        )
        instructions.pack(fill='x', pady=(0, 8))

        # Text Area Outer Border Box
        text_border = tk.Frame(card_inner, bg='#CBD5E1', bd=1)
        text_border.pack(fill='both', expand=True, pady=(0, 15))

        self.email_text = tk.Text(
            text_border, 
            height=8, 
            wrap='word', 
            bg='#FFFFFF', 
            fg=self.TEXT_MAIN, 
            font=("Segoe UI", 10),
            relief='flat',
            padx=10,
            pady=10,
            insertbackground=self.TEXT_MAIN
        )
        self.email_text.pack(fill='both', expand=True)

        # Action Buttons
        btn_frame = tk.Frame(card_inner, bg=self.CARD_BG)
        btn_frame.pack(fill='x', pady=(0, 15))

        self.submit_button = tk.Button(
            btn_frame, 
            text="🔍 Check Message", 
            command=self.process_email, 
            font=("Segoe UI", 11, "bold"), 
            bg=self.PRIMARY, 
            fg='#FFFFFF', 
            activebackground=self.PRIMARY_HOVER,
            activeforeground='#FFFFFF',
            relief='flat', 
            padx=16, 
            pady=8,
            cursor='hand2'
        )
        self.submit_button.pack(side='left', padx=(0, 10))

        self.delete_button = tk.Button(
            btn_frame, 
            text="🗑️ Clear Text", 
            command=self.delete_mail, 
            font=("Segoe UI", 10), 
            bg='#F1F5F9', 
            fg=self.TEXT_MAIN, 
            activebackground='#E2E8F0',
            relief='flat', 
            padx=14, 
            pady=8,
            cursor='hand2'
        )
        self.delete_button.pack(side='left')

        # Status / Result Dynamic Badge Box
        self.result_card = tk.Frame(card_inner, bg='#F1F5F9', padx=15, pady=12)
        self.result_card.pack(fill='x')

        self.status_label = tk.Label(
            self.result_card, 
            text="Ready to scan. Paste text above and click Check.", 
            font=("Segoe UI", 10), 
            bg='#F1F5F9', 
            fg=self.TEXT_MUTED,
            wraplength=480,
            justify='left'
        )
        self.status_label.pack(anchor='w')

        # Footer
        footer = tk.Frame(self.root, bg=self.BG_COLOR, pady=12)
        footer.pack(side='bottom', fill='x')
        footer_text = tk.Label(
            footer, 
            text="© 2026 Spam Filter AI • Powered by Naive Bayes & NLP", 
            font=("Segoe UI", 9), 
            fg=self.TEXT_MUTED, 
            bg=self.BG_COLOR
        )
        footer_text.pack()

    def _load_artifacts(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        except FileNotFoundError:
            self.submit_button.config(state='disabled')
            messagebox.showerror(
                "Model Files Missing",
                "Could not find model files. Run these commands from project root:\n"
                "1) python src/data_preprocessing.py\n"
                "2) python src/feature_extraction.py\n"
                "3) python src/model.py"
            )

    def process_email(self):
        if self.model is None or self.vectorizer is None:
            messagebox.showerror("Model Not Loaded", "Model is not loaded yet. Please generate model files first.")
            return

        email_content = self.email_text.get("1.0", tk.END).strip()
        if email_content:
            preprocessed_content = preprocess_text(email_content)
            features = self.vectorizer.transform([preprocessed_content])
            prediction = self.model.predict(features)

            if prediction == 1:
                self.result_card.config(bg='#FEE2E2')
                self.status_label.config(
                    text="🚨 SPAM DETECTED — Exercise caution with this email.", 
                    fg='#DC2626', 
                    bg='#FEE2E2',
                    font=("Segoe UI", 10, "bold")
                )
            else:
                self.result_card.config(bg='#D1FAE5')
                self.status_label.config(
                    text="✅ LEGITIMATE EMAIL — No spam patterns detected.", 
                    fg='#059669', 
                    bg='#D1FAE5',
                    font=("Segoe UI", 10, "bold")
                )
        else:
            messagebox.showwarning("Empty Content", "Please paste an email into the text area before submitting.")

    def delete_mail(self):
        self.email_text.delete("1.0", tk.END)
        self.result_card.config(bg='#F1F5F9')
        self.status_label.config(
            text="Ready to scan. Paste text above and click Check.", 
            fg=self.TEXT_MUTED, 
            bg='#F1F5F9',
            font=("Segoe UI", 10)
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamFilterAIApp(root)
    root.mainloop()