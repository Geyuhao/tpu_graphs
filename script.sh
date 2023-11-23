python tiles_train.py --model=EarlyJoinResGCN
python tiles_train.py --model=LateJoinResGCN
python tiles_train.py --model=EarlyJoinSAGE
python tiles_train.py --model=LateJoinSAGE
python tiles_train.py --model=MLP


for dir in ~/out/model_*; do
    python tiles_evaluate.py --dirs "$dir"
done