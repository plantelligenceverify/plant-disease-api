# Plant Disease Detection API

## Folder Structure
```
plant_disease_api/
├── main.py
├── requirements.txt
├── render.yaml
├── README.md
└── model/
    ├── mobilenetv2_plantcity.tflite  ← apna model yahan rakho
```

## Render.com pe Deploy Karo (FREE)

1. GitHub pe repo banao
2. Ye sab files push karo
3. model/ folder mein tflite file rakho
4. render.com pe jao → New Web Service
5. GitHub repo connect karo
6. Deploy!

## API Endpoints

### Health Check
GET /
Response: {"status": "running", "classes": 52}

### Detect Disease
POST /detect
Body: {"image_url": "https://...", "scan_id": "123"}

### Webhook (Supabase se auto)
POST /webhook
