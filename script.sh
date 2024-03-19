# #!/bin/bash

# # EarlyJoinSAGEAGG config = 5 bad result batch = 10 bad result
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 5 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG540.log 2>&1
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 10 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG1010.log 2>&1

# # change loss hinge loss bad result
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 40 --losses PairwiseHingeLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGHinge.log 2>&1
# echo "EarlyJoinSAGEAGG Done"

# # python tiles_train.py --model=EarlyJoinSAGEAGGBI --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGBI.log 2>&1

# # EarlyJoinSAGEAGGRes res no improvement
# python tiles_train.py --model=EarlyJoinSAGEAGGRes --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGRes.log 2>&1
# echo "EarlyJoinSAGEAGGRes Done"

# # EarlyJoinSAGEAGG2 no drop out better than EarlyJoinSAGEAGG
# python tiles_train.py --model=EarlyJoinSAGEAGG2 --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG2.log 2>&1

# echo "EarlyJoinSAGEAGG2 Done"

# # EarlyJoinSAGEAGG no drop out 5 gnn layer
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG5.log 2>&1

# new EarlyJoinSAGEAGGRes
# python tiles_train.py --model=EarlyJoinSAGEAGGRes --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGRes2.log 2>&1

# # new EarlyJoinSAGEAGG2 new dropout position
# python tiles_train.py --model=EarlyJoinSAGEAGG2 --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG2.log 2>&1

# EarlyJoinSAGEAGG large batch size
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 100 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG.log 2>&1



# ## below are script after know the current best combination: AGG + dropout
# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG.log 2>&1
# echo "EarlyJoinSAGEAGG Done"

# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 20 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG2040.log 2>&1
# echo "EarlyJoinSAGEAGG2040 Done"

# python tiles_train.py --model=EarlyJoinSAGEAGG --epochs 200 --configs 10 --batch 100 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGG10100.log 2>&1
# echo "EarlyJoinSAGEAGG10100 Done"

# # add residual connection
# python tiles_train.py --model=EarlyJoinSAGEAGGRes --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGRes.log 2>&1
# echo "EarlyJoinSAGEAGGRes Done"

# # use feature embedding EarlyJoinSAGEAGGSplit
# python tiles_train.py --model=EarlyJoinSAGEAGGSplit --epochs 200 --configs 10 --batch 40 --losses PairwiseLogisticLoss:1 --lr 0.000642 --clip_norm 1e9 > train_EarlyJoinSAGEAGGSplit.log 2>&1
# echo "EarlyJoinSAGEAGGSplit Done"

# python tiles_gen_predictions_csv_validation.py --hash model --dir ~/out/tpugraphs_tiles/

# python tiles_evaluate_csv.py --name J

# baseline: LateJoinResGCN EarlyJoinResGCN LateJoinSAGE EarlyJoinSAGE 
python tiles_train.py --model=LateJoinResGCN > LateJoinResGCN.log 2>&1
echo "LateJoinResGCN Done"
python tiles_train.py --model=EarlyJoinResGCN > EarlyJoinResGCN.log 2>&1
echo "EarlyJoinResGCN Done"
python tiles_train.py --model=LateJoinSAGE > LateJoinSAGE.log 2>&1
echo "LateJoinSAGE Done"
python tiles_train.py --model=EarlyJoinSAGE > EarlyJoinSAGE.log 2>&1
echo "EarlyJoinSAGE Done"
