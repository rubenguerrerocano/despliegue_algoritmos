from fastapi import FastAPI, HTTPException
from transformers import pipeline

app = FastAPI()

# 1.Saludo
@app.get('/saluda')
def saluda(nombre: str, edad: int):
    return {'Message': f'Hola {nombre}, tienes {edad} años.'}

# 2.Operación matemática
@app.get('/operacion')
def operacion(num1: float, num2: float, operador: str):
    if operador == '+':
        resultado = num1 + num2
    elif operador == '-':
        resultado = num1 - num2
    elif operador == '*':
        resultado = num1 * num2
    elif operador == '/':
        if num2 == 0:
            raise HTTPException(status_code=400, detail="División por cero no permitida")
        resultado = num1 / num2
    else:
        raise HTTPException(status_code=400, detail="Operador no válido. Usa +, -, *, /")
    
    return {'Resultado': resultado}

# 3️.Análisis de sentimiento (HF)
@app.get('/sentiment')
def sentiment_classification(text: str, modelo: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    sentiment_pipeline = pipeline('sentiment-analysis', model=modelo)
    resultado = sentiment_pipeline(text)[0]
    return {'Sentiment': resultado['label'], 'Confidence': resultado['score']}

# 4.Resumen de texto (HF)
@app.get('/resumen')
def resumen_texto(text: str, max_length: int = 50):
    summarizer = pipeline("summarization")
    resultado = summarizer(text, max_length=max_length, min_length=10, do_sample=False)
    return {'Resumen': resultado[0]['summary_text']}

# 5. Traducción español-inglés (HF)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

@app.get('/traducir')
def traducir(texto: str):
    try:
        resultado = translator(texto)[0]['translation_text']
        return {'Traducción': resultado}
    except Exception as e:
        return {'error': str(e)}
