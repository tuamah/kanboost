# خارطة طريق مشروع KANBoost (Gradient Boosting with KAN)

**الهدف:** بناء مكتبة مفتوحة المصدر لتعلّم الآلة تستخدم KAN كـ weak learner بمنطق Gradient Boosting، تنافس CatBoost/XGBoost/LightGBM بالدقة، وتتفوق عليهم بالتفسير الرياضي (interpretability).

**الترخيص المقترح:** MIT License (أوسع انتشار، يشجع التبني والمساهمة)

---

## المرحلة 0: الأساس البحثي (2-3 أسابيع)

- [ ] قراءة كاملة لورقة **GB-KAN** (ICAART 2026) — هي أقرب عمل موجود لفكرتك، لازم تفهمها بعمق قبل أي كود
- [ ] قراءة **KAN or MLP: A Fairer Comparison** (نقدية، لازم تعرف نقاط ضعف KAN بصدق)
- [ ] قراءة ورقة KAN الأصلية (Liu et al. 2024) + KAN 2.0
- [ ] مسح المكتبات الموجودة: pykan, efficient-kan, FastKAN — تحديد أيها أساس أفضل للبناء عليه (الأرجح efficient-kan أو FastKAN للسرعة)
- [ ] التأكد عمليًا: هل GB-KAN لها كود منشور علنًا؟ (لو لا، هذي فرصتك الحقيقية لأول تطبيق مفتوح)

**الفجوة البحثية التي تسدها:** لا توجد دراسة منشورة تقيّم GB-KAN على بيانات تسويقية/churn ضخمة (100K+ صف) مقابل CatBoost تحديدًا.

---

## المرحلة 1: النموذج الأولي (Prototype) — 3-4 أسابيع

### 1.1 التصميم الخوارزمي
```
خوارزمية GB-KAN (Gradient Boosting):
1. ابدأ بتنبؤ ثابت F0(x) = متوسط y
2. لكل تكرار t من 1 إلى T:
   a. احسب pseudo-residuals: r_i = -∂Loss/∂F(x_i)
   b. درّب KAN صغيرة (weak learner) ft على (X, r)
   c. حدّث: F(t)(x) = F(t-1)(x) + ν · ft(x)   [ν = learning rate]
3. التنبؤ النهائي = F(T)(x)
```

### 1.2 قرارات تصميم أساسية يجب اتخاذها بالكود
- [ ] حجم KAN الفردية: width=[n_features, k, 1] — k صغير جدًا (2-4) لضمان السرعة كـ "weak" learner
- [ ] grid منخفض (2-3) لكل learner فردي
- [ ] عدد التكرارات (n_estimators): 50-300 كبداية
- [ ] معالجة Loss functions متعددة: BCE (تصنيف)، MSE (انحدار)
- [ ] معالجة Missing values: بناء imputer مدمج (نقطة ضعف KAN الحالية مقابل CatBoost)
- [ ] معالجة Categorical: ترميز مدمج تلقائي (target encoding أو embedding بسيط) بدل one-hot يدوي

### 1.3 الكود الأساسي (نقطة انطلاق)
```python
import numpy as np
from kan import KAN
import torch

class GBKANClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 kan_width=3, kan_grid=2):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.kan_width = kan_width
        self.kan_grid = kan_grid
        self.learners = []
        self.init_pred = None

    def fit(self, X, y):
        n_features = X.shape[1]
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # التنبؤ الابتدائي (log-odds للتصنيف الثنائي)
        p = y.mean()
        self.init_pred = np.log(p / (1 - p))
        F = np.full(len(y), self.init_pred)

        for t in range(self.n_estimators):
            prob = 1 / (1 + np.exp(-F))
            residual = y - prob   # pseudo-residual لـ Logloss

            learner = KAN(width=[n_features, self.kan_width, 1],
                          grid=self.kan_grid, k=3, seed=t)
            r_t = torch.tensor(residual, dtype=torch.float32).unsqueeze(1)
            dataset = {'train_input': X_t, 'train_label': r_t,
                       'test_input': X_t, 'test_label': r_t}
            learner.fit(dataset, opt="Adam", steps=20, lr=0.01)

            with torch.no_grad():
                update = learner(X_t).numpy().flatten()
            F += self.lr * update
            self.learners.append(learner)

        return self

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        F = np.full(len(X), self.init_pred)
        for learner in self.learners:
            with torch.no_grad():
                F += self.lr * learner(X_t).numpy().flatten()
        return 1 / (1 + np.exp(-F))
```

### 1.4 اختبار أولي
- [ ] تجربة على بيانات churn عندك (نفس التي استخدمناها) — قياس AUC مقابل CatBoost (0.6992)
- [ ] تجربة على 2-3 بيانات UCI معيارية (Breast Cancer, Adult Income, California Housing) للمقارنة المباشرة مع نتائج ورقة GB-KAN المنشورة

---

## المرحلة 2: التحسين الهندسي (6-8 أسابيع) — الأصعب

- [ ] **السرعة**: قياس زمن التدريب الحالي، ثم:
  - تجربة `torch.compile` لتسريع كل KAN فردية
  - تجربة استبدال B-spline بـ FastKAN (RBF) داخل كل weak learner للسرعة
  - Parallelization: تدريب متوازي إن أمكن (Gradient Boosting تسلسلي بطبيعته، لكن كل learner نفسه يقبل batch parallelism)
- [ ] **معالجة تلقائية للفئوية**: بناء encoder مدمج (شبيه بـ CatBoost's ordered target statistics)
- [ ] **Early stopping**: مراقبة validation loss وتوقف تلقائي
- [ ] **Regularization**: دمج L1 على معاملات spline (كما بـ pykan الأصلية) لتفادي overfitting بكل weak learner
- [ ] **Feature importance**: استخراج أهمية الفيتشرز (مجموع مساهمة كل learner لكل فيتشر) — هذي أقوى نقطة تسويقية (تفسير أفضل من CatBoost)
- [ ] **استخراج معادلة رمزية للنموذج الكامل** (لو ممكن) — ميزة فريدة ما عند أي GBDT

---

## المرحلة 3: البنچماركينج الشامل (3-4 أسابيع)

| البيانات | الحجم | المقارنة مع |
|---|---|---|
| churn (عندك) | 100K | CatBoost, XGBoost, LightGBM |
| UCI Adult Income | ~48K | نفس أعلاه |
| California Housing | ~20K | نفس أعلاه (انحدار) |
| Breast Cancer Wisconsin | ~570 | نفس أعلاه (بيانات صغيرة) |
| بيانات صناعية إضافية (اختياري) | - | - |

**المقاييس:** AUC/Accuracy، زمن التدريب، عدد البارامترات، Calibration (ECE)، قابلية التفسير (كمية/نوعية)

---

## المرحلة 4: التوثيق والنشر مفتوح المصدر (2-3 أسابيع)

- [ ] بنية مشروع بايثون قياسية:
```
kanboost/
├── kanboost/
│   ├── __init__.py
│   ├── classifier.py
│   ├── regressor.py
│   ├── encoders.py      # معالجة الفئوية
│   └── utils.py
├── tests/
├── examples/
├── docs/
├── README.md
├── LICENSE (MIT)
├── setup.py / pyproject.toml
└── requirements.txt
```
- [ ] رفع على GitHub بترخيص MIT
- [ ] نشر على PyPI (`pip install kanboost`)
- [ ] README قوي: benchmarks، أمثلة كود، مقارنات رسومية
- [ ] توثيق Sphinx/MkDocs

---

## المرحلة 5: النشر الأكاديمي (بالتوازي، 4-6 أسابيع)

- [ ] كتابة ورقة بحثية: "KANBoost: An Interpretable Gradient Boosting Framework with KAN Learners — Evaluation on Real-World Churn Data"
- [ ] أهداف نشر مناسبة لحجم المشروع: workshop papers (ICAART، AICT نفس مكان GB-KAN)، أو arXiv preprint مباشر
- [ ] تقديم للمؤتمرات الطلابية أو GCI (نفس المنصة اللي استخدمتها بمشروع churn)

---

## المرحلة 6: بناء الانتشار (استمراري)

- [ ] كتابة مقالات (Medium/Dev.to/LinkedIn) تشرح الفكرة والنتائج
- [ ] نشر على Reddit (r/MachineLearning) وHacker News عند الجهوزية
- [ ] التواصل مع فريق pykan الأصلي / مؤلفي ورقة GB-KAN (تواصل أكاديمي، ممكن تعاون)
- [ ] فتح المشروع لمساهمات (good first issues، CONTRIBUTING.md)

---

## المخاطر والتحديات الصادقة

| الخطر | التخفيف |
|---|---|
| GB-KAN لا تنافس CatBoost بالسرعة أبداً بدون C++/CUDA حقيقي | ركّز على قيمة "التفسير" كميزة تنافسية أساسية، مش السرعة |
| نتائج مخيبة على بيانات كبيرة (كما شفنا بتجربتنا) | كن صادق بالنتائج بالورقة/التوثيق — النقد العلمي المتوازن يبني مصداقية أكبر من التسويق المبالغ |
| مجال بحثي متغير بسرعة (قد يظهر بديل أقوى قبل ما تنتهي) | انشر مبكرًا (MVP + benchmark أولي) بدل الانتظار للكمال |

---

## الجدول الزمني التقديري الكامل

| المرحلة | المدة | تراكمي |
|---|---|---|
| 0. الأساس البحثي | 2-3 أسابيع | 3 أسابيع |
| 1. النموذج الأولي | 3-4 أسابيع | 7 أسابيع |
| 2. التحسين الهندسي | 6-8 أسابيع | 15 أسبوع |
| 3. البنچماركينج | 3-4 أسابيع | 19 أسبوع |
| 4. النشر مفتوح المصدر | 2-3 أسابيع | 22 أسبوع |
| 5. النشر الأكاديمي | 4-6 أسابيع (متوازي) | ~24-26 أسبوع |

**الإجمالي: تقريباً 6 أشهر لمشروع كامل بمعايير أكاديمية وتقنية جيدة.**
