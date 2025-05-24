run-dev:
	uvicorn predict_service.main:app --host 0.0.0.0 --port 8004 --reload;

train:
	python predict_service/ml/wealth_cost_prediction.py;

run-front-dev:
	streamlit run streamlit-insurance-app/src/app.py