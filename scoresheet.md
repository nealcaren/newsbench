# NewsBench score sheet

| model | harness | region | overall | cased | broad | page | chrF | bowF1 | gap | $ | $/100pg | out_tok | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-ocr-doclayout | newspaper-ocr | DocLayout | 0.970 | 0.955 | 0.976 | 0.967 | 0.986 | 0.986 | 0.015 | 0.0000 | 0.00 |  | 15 |
| glm-ocr-residual | newspaper-ocr | PaddleX | 0.937 | 0.922 | 0.964 | 0.919 | 0.979 | 0.980 | 0.043 | 0.0000 | 0.00 |  | 15 |
| gemini-3.5-flash-lite-residual | newspaper-ocr | PaddleX | 0.936 | 0.915 | 0.967 | 0.915 | 0.972 | 0.944 | 0.008 | 0.8577 | 4.51 | 197117 | 15 |
| progmag-ppdoc-glm-repair | progmag-ocr | PaddleX | 0.924 | 0.910 | 0.973 | 0.891 | 0.956 | 0.965 | 0.041 | 0.0000 | 0.00 |  | 15 |
| gpt-5.6-luna | newspaper-ocr | PaddleX | 0.921 | 0.898 | 0.974 | 0.885 | 0.941 | 0.928 | 0.007 | 0.0002 | 0.00 | 723904 | 15 |
| deepseek-v4.1-flash | newspaper-ocr | PaddleX | 0.920 | 0.899 | 0.972 | 0.886 | 0.940 | 0.930 | 0.009 | 0.5181 | 2.73 | 565761 | 15 |
| gemini-3.5-flash-lite | newspaper-ocr | PaddleX | 0.920 | 0.899 | 0.973 | 0.885 | 0.941 | 0.929 | 0.009 | 0.6471 | 3.41 | 124801 | 15 |
| lib-chunk | newspaper-ocr | PaddleX | 0.919 | 0.905 | 0.969 | 0.886 | 0.947 | 0.961 | 0.043 | 0.0000 | 0.00 |  | 15 |
| lib-paddlex-glm | newspaper-ocr | PaddleX | 0.919 | 0.905 | 0.969 | 0.886 | 0.947 | 0.961 | 0.043 | 0.0000 | 0.00 |  | 15 |
| paddlex-glm | newspaper-ocr | PaddleX | 0.919 | 0.905 | 0.969 | 0.886 | 0.947 | 0.961 | 0.043 | 0.0000 | 0.00 |  | 15 |
| res-glm-base | newspaper-ocr | PaddleX | 0.919 | 0.905 | 0.969 | 0.886 | 0.947 | 0.961 | 0.043 | 0.0000 | 0.00 |  | 15 |
| mistral-small-2603 | newspaper-ocr | PaddleX | 0.917 | 0.899 | 0.971 | 0.881 | 0.944 | 0.944 | 0.027 | 0.1343 | 0.71 | 118448 | 15 |
| res-tess-resid | newspaper-ocr | PaddleX | 0.917 | 0.890 | 0.958 | 0.889 | 0.949 | 0.905 | -0.011 | 0.0000 | 0.00 |  | 15 |
| glm-5.3-flash | newspaper-ocr | PaddleX | 0.916 | 0.901 | 0.971 | 0.880 | 0.943 | 0.940 | 0.023 | 0.1318 | 0.69 | 218219 | 15 |
| lib-fallback | newspaper-ocr | PaddleX | 0.915 | 0.901 | 0.969 | 0.880 | 0.947 | 0.960 | 0.045 | 0.0000 | 0.00 |  | 15 |
| progmag-ppdoc-tesseract | progmag-ocr | PaddleX | 0.914 | 0.890 | 0.967 | 0.880 | 0.930 | 0.902 | -0.013 | 0.0000 | 0.00 |  | 15 |
| paddlex-glm-repair | newspaper-ocr | PaddleX | 0.913 | 0.899 | 0.962 | 0.881 | 0.945 | 0.959 | 0.045 | 0.0000 | 0.00 |  | 15 |
| qwen3.8-flash | newspaper-ocr | PaddleX | 0.907 | 0.887 | 0.965 | 0.868 | 0.931 | 0.924 | 0.017 | 0.1545 | 0.81 | 215766 | 15 |
| lib-repair | newspaper-ocr | PaddleX | 0.906 | 0.892 | 0.961 | 0.869 | 0.939 | 0.954 | 0.048 | 0.0000 | 0.00 |  | 15 |
| paddlex-tesseract | newspaper-ocr | PaddleX | 0.899 | 0.874 | 0.962 | 0.857 | 0.920 | 0.891 | -0.008 | 0.0000 | 0.00 |  | 15 |
| res-tess-base | newspaper-ocr | PaddleX | 0.899 | 0.874 | 0.962 | 0.857 | 0.920 | 0.891 | -0.008 | 0.0000 | 0.00 |  | 15 |
| claude-haiku-4.5 | newspaper-ocr | PaddleX | 0.894 | 0.874 | 0.964 | 0.848 | 0.937 | 0.917 | 0.023 | 1.0645 | 5.60 | 134382 | 15 |
| gemma-3-27b-it | newspaper-ocr | PaddleX | 0.875 | 0.852 | 0.953 | 0.823 | 0.917 | 0.906 | 0.031 | 0.0627 | 0.33 | 124086 | 15 |
| gemini-3.5-flash-lite | newspaper-ocr | none | 0.820 | 0.803 | 0.884 | 0.778 | 0.876 | 0.867 | 0.048 | 0.5477 | 2.88 | 217140 | 15 |
| tesseract | newspaper-ocr | none | 0.677 | 0.662 | 0.614 | 0.719 | 0.866 | 0.844 | 0.167 | 0.0000 | 0.00 |  | 15 |
| tesseract-default | newspaper-ocr | AS-YOLO | 0.622 | 0.604 | 0.608 | 0.631 | 0.686 | 0.710 | 0.089 | 0.0000 | 0.00 |  | 15 |
| news_combo_fast | newspaper-ocr | AS-YOLO | 0.617 | 0.601 | 0.603 | 0.627 | 0.669 | 0.711 | 0.094 | 0.0000 | 0.00 |  | 15 |
| glm-ocr-mlx | newspaper-ocr | AS-YOLO | 0.607 | 0.596 | 0.574 | 0.629 | 0.671 | 0.750 | 0.143 | 0.0000 | 0.00 |  | 15 |
| gpt-5.6-luna | newspaper-ocr | none | 0.503 | 0.490 | 0.378 | 0.587 | 0.690 | 0.730 | 0.226 | 0.0000 | 0.00 | 133143 | 15 |
| qwen3.8-flash | newspaper-ocr | none | 0.341 | 0.332 | 0.185 | 0.446 | 0.349 | 0.362 | 0.021 | 0.0279 | 0.15 | 80352 | 15 |
| lib-paddlex-glm-raw | newspaper-ocr | PaddleX | 0.291 | 0.293 | 0.253 | 0.317 | 0.946 | 0.952 | 0.660 | 0.0000 | 0.00 |  | 15 |
| claude-haiku-4.5 | newspaper-ocr | none | 0.232 | 0.231 | 0.165 | 0.276 | 0.360 | 0.397 | 0.165 | 0.3060 | 1.61 | 53375 | 15 |
| gemma-3-27b-it | newspaper-ocr | none | 0.226 | 0.223 | 0.187 | 0.252 | 0.339 | 0.356 | 0.130 | 0.0270 | 0.14 | 62568 | 15 |
| mistral-small-2603 | newspaper-ocr | none | 0.202 | 0.199 | 0.118 | 0.258 | 0.314 | 0.358 | 0.156 | 0.0913 | 0.48 | 112627 | 15 |
