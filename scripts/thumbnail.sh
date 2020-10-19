#!/bin/bash
FILES= "/media/matt/New\ Volume/ava/ava-compressed/images/*.jpg"

mogrify -format gif -path thumbnails -thumbnail 224x224 $FILES
