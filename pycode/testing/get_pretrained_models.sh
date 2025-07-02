#!/usr/bin/env sh

cd $(dirname "$0")

mkdir -p pretrained_models
cd pretrained_models

for item in \
'v1.1/float_model_3classes_TC_PN_2_3_8_explicit_norm_4.h5' \
'v1.1/float_model_5classes_TC_PN_2_3_8_explicit_norm_2.h5' \
'v2.0/float_model_5classes_TC_PN_4_3_32_explicit_norm_2.h5'; do
  mkdir -p "$(dirname "$item")"
  curl -fsSLo "$item" "https://ipaccess.ehu.eus/HSI-Drive/files/pretrained_models/$item"
done

