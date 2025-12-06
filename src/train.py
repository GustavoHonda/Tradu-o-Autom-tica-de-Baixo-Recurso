from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir="./mt5-finetuned",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()
    trainer.save_model("./model/mt5-finetuned")
    tokenizer.save_pretrained("./model/mt5-finetuned")
