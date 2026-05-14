MLDS_IDS│
├── 📁 data/
│   ├── 📁 raw_pcaps/                 # PCAP files go here
│   │   └── capture_*.pcap
│   │
│   └── 📁 processed_flows/           # Generated output files
│       ├── flows.csv                 # Raw flow data
│       ├── features.csv              # 52 extracted features
│       ├── features_scaled.csv       # Scaled features
│       ├── rule_predictions.csv      # Rule-based results
│       ├── rf_predictions.csv        # Random Forest results
│       ├── xgb_predictions.csv       # XGBoost results
│       ├── et_predictions.csv        # Extra Trees results
│       ├── dt_predictions.csv        # Decision Tree results
│       ├── gb_predictions.csv        # Gradient Boosting results
│       ├── ensemble_predictions.csv  # Ensemble (majority vote)
│       └── processing_summary.csv    # Processing history
│
├── 📁 model/
│   ├── rf_model.pkl                  # Random Forest
│   ├── xgb_model.json                # XGBoost
│   ├── et_model.pkl                  # Extra Trees
│   ├── dt_model.pkl                  # Decision Tree
│   ├── gb_model.joblib               # Gradient Boosting
│   └── label_mapping.json            # Attack type mapping
│
├── 📁 src/
│   ├── base_predictor.py             # Base class for all predictors
│   ├── rf_predictor.py               # Random Forest wrapper
│   ├── xgb_predictor.py              # XGBoost wrapper
│   ├── et_predictor.py               # Extra Trees wrapper
│   ├── dt_predictor.py               # Decision Tree wrapper
│   ├── gb_predictor.py               # Gradient Boosting wrapper
│   ├── rule_based_predictor.py       # Rule-based detection
│   ├── flow_generator.py             # TShark flow extraction
│   ├── feature_extractor.py          # 52 feature extraction
│   ├── feature_scaler.py             # Feature scaling
│   └── gui_auto.py                   # Streamlit dashboard
│
├── auto_processor.py                 # Main processing script
├── controller.py                     # Process orchestrator
├── fully_automated.py                # Interactive menu
├── continuous_capture.py             # Live packet capture
├── Start_Control.bat                 # Windows launcher
└── requirements.txt                  # Python dependencies
