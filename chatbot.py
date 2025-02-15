import os
import numpy as np
import torch
import faiss
from datasets import Dataset
import fitz
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


class Chatbot:
    """This class represents the behavior of a chatbot using
    both fine-tuning and RAG to adjust it."""

    def __init__(self):
        self.fresh_data = ["docs/informacion_actual.pdf"]
        self.model_save_path = "./results/model"
        model_name = "distilbert-base-uncased"
        gpt_model_name = "gpt2"

        self.tokenizer = self._generate_tokenizer(model_name)
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)

        if self.gpt_tokenizer.pad_token is None:
            self.gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.gpt_tokenizer.eos_token is None:
            self.gpt_tokenizer.add_special_tokens({'eos_token': '[EOS]'})

        if os.path.exists(self.model_save_path):
            print("Load trained model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_save_path
            )
        else:
            print("Training new model...")
            concepts_path = "docs/informacion_general.pdf"
            self.concepts = self._generate_dataset(concepts_path)
            self.model = self._load_foundational_model(model_name)
            self._fine_tuning()

        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.index = self._load_fresh_data(self.fresh_data)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def _embed(self, text: str) -> np.ndarray:
        """
        Computes a 768-dimensional embedding for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedding vector.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling on the token embeddings.
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings.cpu().numpy()[0]

    def _load_foundational_model(
        self, model_name: str
    ) -> AutoModelForSequenceClassification:
        """Loads the foundational model to fine-tune it.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            The model.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True
        )
        return model

    def _generate_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Generates the tokenizer for the model.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            The tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    def _generate_dataset(self, path_concepts: str) -> Dataset:
        """Generates a dataset from the given PDF file.

        Args:
            path_concepts (str): The path to the PDF file.

        Returns:
            Dataset: The generated dataset.
        """
        text_concepts = self._extract_text_from_pdf(path_concepts)
        dataset = Dataset.from_dict({"text": [text_concepts], "labels": [0]})
        return dataset

    def _tokenize_function(self, examples: dict) -> dict:
        """Tokenizes the examples for the dataset.

        Args:
            examples (dict): The examples to tokenize.

        Returns:
            dict: The tokenized examples.
        """
        outputs = self.tokenizer(
            examples["text"], padding="max_length", truncation=True
        )
        # Pass through the labels so the model can compute the loss
        outputs["labels"] = examples["labels"]
        return outputs

    def _fine_tuning(self):
        """Fine-tunes the model using the generated dataset."""
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )

        tokenized_dataset = self.concepts.map(self._tokenize_function, batched=True)

        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=tokenized_dataset
        )

        trainer.train()
        trainer.save_model(self.model_save_path)

    def _load_fresh_data(self, documents: list):
        """Loads fresh data and creates an index for retrieval.

        Args:
            documents (list): The list of document paths.

        Returns:
            faiss.IndexFlatL2: The created index.
        """
        index = faiss.IndexFlatL2(768)
        embeddings = np.vstack(
            [self._embed(self._extract_text_from_pdf(doc)) for doc in documents]
        ).astype(np.float32)
        index.add(embeddings)
        return index

    def _retrieve(self, query: str, documents: list):
        """Retrieves the most similar document for the given query.

        Args:
            query (str): The query text.
            documents (list): The list of document paths.

        Returns:
            list: The list of retrieved documents.
        """
        query_embedding = self._embed(query)
        _, i_ = self.index.search(np.array([query_embedding]), 1)
        return [documents[i] for i in i_[0]]

    def generate_response(self, prompt: str) -> str:
        """Generates a response for a given prompt.

        Args:
            prompt (str): The prompt to generate the response.

        Returns:
            str: The generated response.
        """
        retrieved_document = self._retrieve(prompt, self.fresh_data)
        context = " ".join(retrieved_document)
        input_text = f"{context} {prompt}"
        inputs = self.gpt_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        output = self.gpt_model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Provide the attention mask
            max_length=150, 
            num_return_sequences=1, 
            temperature=0.7,  # Control randomness
            top_k=50,         # Control diversity
            top_p=0.9,        # Control diversity
            repetition_penalty=2.0,  # Penalize repetition
            pad_token_id=self.gpt_tokenizer.eos_token_id,  # Set the pad_token_id
            do_sample=True  # Enable sampling
        )
        return self.gpt_tokenizer.decode(output[0], skip_special_tokens=True)