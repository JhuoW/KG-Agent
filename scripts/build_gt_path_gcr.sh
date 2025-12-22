DATA_PATH="RoG-webqsp RoG-cwq"
SPLIT=train
N_PROCESS=64
for DATA_PATH in ${DATA_PATH}; do
  python build_gt_path_gcr.py --d ${DATA_PATH} --split ${SPLIT} --n ${N_PROCESS}
done