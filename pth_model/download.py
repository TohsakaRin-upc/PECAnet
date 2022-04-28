# -*- coding: utf-8 -*-

import wget

DATA_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

wget.download(DATA_URL)
