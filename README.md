# 딥러닝 A to Z · 초파리 특별편

**언어 · Language:** [한국어](#ko) · [English](#english) · [中文](#zh) · [日本語](#ja) · [Oʻzbekcha](#ozbekcha) · [Русский](#ru)

| 언어 · Language | 첫 화면 · Home |
|---|---|
| 한국어 (기본 · default) | [index.html](index.html) |
| English | [en/index.html](en/index.html) |
| 中文（简体） | [zh/index.html](zh/index.html) |
| 日本語 | [ja/index.html](ja/index.html) |
| Oʻzbekcha | [uz/index.html](uz/index.html) |
| Русский | [ru/index.html](ru/index.html) |

---

<a id="ko"></a>

## 한국어

초파리 뇌 연구와 인공 신경망을 나란히 놓고, 버튼을 눌러 그림과 숫자를 바꿔 보며 배우는 교육용 웹 페이지 세 편입니다.

글 · 안상선 ((주)M-Robo 대표)

### 편 목록

| 편 | 파일 | 내용 |
|---|---|---|
| 처음 화면 | [index.html](index.html) | 연구실 식구들, 연구 지도, 보는 법 |
| 특별 1 | [1_fruitfly_brain_deep_learning.html](1_fruitfly_brain_deep_learning.html) | 초파리 뇌 탐사 — 감각에서 인식까지, 신경망이 한 칸씩 커지는 과정 |
| 특별 2 | [2_fruitfly_memory_retrieval_paper.html](2_fruitfly_memory_retrieval_paper.html) | 배운 기억은 본능 회로를 거쳐서 꺼내진다 — Dolan 외 (2018) *Neuron* 도해 |
| 특별 3 | [3_fruitfly_minecraft_connectome.html](3_fruitfly_minecraft_connectome.html) | 마인크래프트 속 초파리 뇌 — 배선도만으로 움직이는 신경계 |
| 교육자용 | [answers.html](answers.html) | 확인 문제 정답, 채점 기준, Colab 과제 예시 답안 |

각 편에는 공통 섹션 A~F(Colab 코드 2개, AI 응용 프롬프트, 용어 정리, 비유 모음, 저자 소개와 레퍼런스, 교육자 TIP)와 마무리 장면이 있습니다. 코드 1(넘파이)과 코드 2(PyTorch)는 각 편 공통 섹션 A에서 복사하거나 내려받습니다.

### 다국어 버전

- 기본은 한국어이며, 같은 내용을 **영어 · 중국어 · 일본어 · 우즈베크어 · 러시아어**로 옮긴 사본이 `en/` `zh/` `ja/` `uz/` `ru/` 폴더에 있습니다. 그림, 버튼 동작, 용어 말풍선, 퀴즈, Colab 코드의 주석과 출력 문구까지 모두 번역되어 있습니다.
- **자동 전환:** 한국어 페이지를 처음 열면 브라우저 언어를 보고 지원하는 언어(영어·중국어·일본어·우즈베크어·러시아어)의 같은 페이지로 자동 이동합니다. 지원하지 않는 언어면 한국어를 그대로 보여 줍니다.
- **언어 선택:** 모든 페이지의 **최상단 메뉴**에 🌐 언어 선택 메뉴가 있고, 맨 위 **언어 배너**에서도 바로 바꿀 수 있습니다. 한 번 고른 언어는 브라우저가 기억해 다음 방문에도 그 언어로 열립니다. 배너의 ×는 이번 방문 동안만 배너를 숨깁니다.
- 링크 끝에 `?lang=stay`를 붙이면 자동 전환 없이 그 페이지를 엽니다.

### 보는 법

- **GitHub Pages:** Settings → Pages → Branch `main`, 폴더 `/ (root)` → Save. 1~3분 뒤 `https://<아이디>.github.io/<저장소 이름>/`에서 첫 화면이 열립니다.
- **내 컴퓨터:** HTML 파일을 브라우저로 엽니다. 그림과 코드가 모두 파일 안에 들어 있어 파일 하나만 열어도 보입니다. 편 사이를 오가거나 언어를 바꾸려면 폴더 구조를 그대로 둡니다.
- **Colab:** 각 편 공통 섹션 A의 "노트북(.ipynb) 내려받기"를 누른 뒤 Colab에서 파일 → 노트북 업로드로 엽니다.

외부에서 불러오는 것은 글꼴(Google Fonts의 IBM Plex Sans KR, jsDelivr의 Galmuri)뿐입니다. 오프라인이면 시스템 글꼴로 바뀌고 나머지는 그대로 동작합니다.

### 라이선스

이 저작물은 [크리에이티브 커먼즈 저작자표시-비영리 4.0 국제 라이선스(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.ko)로 공개합니다. 비상업적 목적에 한해 자유롭게 공유하고 고쳐 쓸 수 있으며, 출처는 반드시 표기해야 합니다. 자세한 내용과 제3자 자료의 표기는 [LICENSE.md](LICENSE.md)에 있습니다.

출처 표기 예: 안상선, 「딥러닝 A to Z · 초파리 특별편」, AI Model Review.

---

<a id="english"></a>

## English

**Deep Learning A to Z · Fruit Fly Special** — three educational web pages that put fruit-fly brain research side by side with artificial neural networks. You learn by pressing buttons and watching pictures and numbers change.

Written by Sangsun Ahn (CEO of M-Robo Inc.)

### Contents

| Part | File | What it covers |
|---|---|---|
| Home | [en/index.html](en/index.html) | Meet the lab, the research map, how to use the pages |
| Special 1 | [en/1_fruitfly_brain_deep_learning.html](en/1_fruitfly_brain_deep_learning.html) | Exploring the fruit fly brain — from sensing to recognition, a neural network that grows one step at a time |
| Special 2 | [en/2_fruitfly_memory_retrieval_paper.html](en/2_fruitfly_memory_retrieval_paper.html) | Learned memories are retrieved through the instinct circuit — an illustrated guide to Dolan et al. (2018) *Neuron* |
| Special 3 | [en/3_fruitfly_minecraft_connectome.html](en/3_fruitfly_minecraft_connectome.html) | A fruit fly brain in Minecraft — a nervous system driven by its wiring diagram alone |
| Educators | [en/answers.html](en/answers.html) | Answers to the check questions, grading criteria, sample solutions for the Colab assignments |

Every part has common sections A–F (two Colab code files, AI prompts, glossary, analogies, author and references, educator TIPs) and a closing scene. Code 1 (NumPy) and Code 2 (PyTorch) can be copied or downloaded from common section A of each part.

### Languages

- Korean is the default. Full translations into **English, Chinese, Japanese, Uzbek and Russian** live in the `en/` `zh/` `ja/` `uz/` `ru/` folders — text, buttons, glossary tooltips, quizzes, and the comments and output messages of the Colab code.
- **Automatic switching:** the first time a Korean page is opened, it checks the browser language and jumps to the same page in that language if it is supported. Otherwise the Korean page stays.
- **Choosing a language:** every page has a 🌐 language menu in its **top menu bar** and a **language banner** at the very top. Your choice is remembered for later visits. The × on the banner hides it for the current visit only.
- Add `?lang=stay` to a link to open that page without automatic switching.

### How to view

- **GitHub Pages:** Settings → Pages → Branch `main`, folder `/ (root)` → Save. After 1–3 minutes the home page opens at `https://<username>.github.io/<repository>/`.
- **On your computer:** open an HTML file in a browser. All pictures and code are embedded, so a single file works on its own. Keep the folder structure to move between parts and languages.
- **Colab:** click "Download notebook (.ipynb)" in common section A, then in Colab use File → Upload notebook.

Only fonts are loaded from outside (IBM Plex Sans KR from Google Fonts, Galmuri from jsDelivr). Offline, system fonts are used and everything else still works.

### License

Released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may share and adapt it for non-commercial purposes; attribution is required. Details and third-party credits are in [LICENSE.md](LICENSE.md).

Attribution example: Sangsun Ahn, "Deep Learning A to Z · Fruit Fly Special", AI Model Review.

---

<a id="zh"></a>

## 中文

**深度学习 A to Z · 果蝇特别篇**——把果蝇大脑研究和人工神经网络放在一起对照的三篇教学网页。按下按钮、改变图和数字，边玩边学。

作者 · Sangsun Ahn（M-Robo 公司代表）

### 篇目

| 篇 | 文件 | 内容 |
|---|---|---|
| 首页 | [zh/index.html](zh/index.html) | 实验室成员、研究地图、使用方法 |
| 特别篇 1 | [zh/1_fruitfly_brain_deep_learning.html](zh/1_fruitfly_brain_deep_learning.html) | 果蝇大脑探索——从感觉到识别，看神经网络一步步变大 |
| 特别篇 2 | [zh/2_fruitfly_memory_retrieval_paper.html](zh/2_fruitfly_memory_retrieval_paper.html) | 学到的记忆要经由本能回路才能被提取——Dolan 等（2018）*Neuron* 图解 |
| 特别篇 3 | [zh/3_fruitfly_minecraft_connectome.html](zh/3_fruitfly_minecraft_connectome.html) | Minecraft 里的果蝇大脑——仅凭接线图运转的神经系统 |
| 教师用 | [zh/answers.html](zh/answers.html) | 练习题答案、评分标准、Colab 作业参考答案 |

每篇都有公共部分 A–F（两段 Colab 代码、AI 应用提示词、术语整理、比喻合集、作者简介与参考文献、教师 TIP）和结尾场景。代码 1（NumPy）和代码 2（PyTorch）可在各篇公共部分 A 中复制或下载。

### 多语言版本

- 默认语言为韩语。`en/` `zh/` `ja/` `uz/` `ru/` 文件夹中是**英语、中文、日语、乌兹别克语、俄语**的完整译本，正文、按钮、术语气泡、测验以及 Colab 代码的注释和输出文字都已翻译。
- **自动切换：**首次打开韩语页面时，会根据浏览器语言自动跳转到对应语言的同一页面；不支持的语言则保持韩语。
- **选择语言：**每个页面的**顶部菜单**都有 🌐 语言选择，页面最上方的**语言横幅**也可直接切换。选过的语言会被浏览器记住。横幅上的 × 只在本次浏览中隐藏横幅。
- 在链接末尾加上 `?lang=stay` 可不经自动切换直接打开该页面。

### 浏览方法

- **GitHub Pages：**Settings → Pages → Branch `main`，文件夹 `/ (root)` → Save。1–3 分钟后在 `https://<用户名>.github.io/<仓库名>/` 打开首页。
- **本地电脑：**用浏览器打开 HTML 文件。图片和代码都内嵌在文件中，单个文件也能显示。若要在各篇或各语言之间切换，请保持文件夹结构不变。
- **Colab：**点击各篇公共部分 A 的“下载笔记本（.ipynb）”，然后在 Colab 中用 文件 → 上传笔记本 打开。

外部加载的只有字体（Google Fonts 的 IBM Plex Sans KR、jsDelivr 的 Galmuri）。离线时改用系统字体，其余功能照常运行。

### 许可

本作品以[知识共享 署名-非商业性使用 4.0 国际许可协议（CC BY-NC 4.0）](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)发布。仅限非商业目的可自由分享和修改，但必须注明出处。详情及第三方资料说明见 [LICENSE.md](LICENSE.md)。

署名示例：Sangsun Ahn，《深度学习 A to Z · 果蝇特别篇》，AI Model Review。

---

<a id="ja"></a>

## 日本語

**ディープラーニング A to Z · ショウジョウバエ特別編** — ショウジョウバエの脳研究と人工ニューラルネットワークを並べて、ボタンを押して図や数字を変えながら学ぶ教育用ウェブページ3編です。

文 · Sangsun Ahn（株式会社M-Robo 代表）

### 編の一覧

| 編 | ファイル | 内容 |
|---|---|---|
| ホーム | [ja/index.html](ja/index.html) | 研究室のメンバー、研究マップ、使い方 |
| 特別編 1 | [ja/1_fruitfly_brain_deep_learning.html](ja/1_fruitfly_brain_deep_learning.html) | ショウジョウバエ脳探検 — 感覚から認識まで、ニューラルネットワークが一段ずつ大きくなる過程 |
| 特別編 2 | [ja/2_fruitfly_memory_retrieval_paper.html](ja/2_fruitfly_memory_retrieval_paper.html) | 学んだ記憶は本能の回路を通って取り出される — Dolan ら（2018）*Neuron* の図解 |
| 特別編 3 | [ja/3_fruitfly_minecraft_connectome.html](ja/3_fruitfly_minecraft_connectome.html) | マインクラフトの中のショウジョウバエ脳 — 配線図だけで動く神経系 |
| 教育者向け | [ja/answers.html](ja/answers.html) | 確認問題の解答、採点基準、Colab 課題の解答例 |

各編には共通セクション A〜F（Colab コード2本、AI 活用プロンプト、用語集、たとえ集、著者紹介と参考文献、教育者向けTIP）と締めくくりの場面があります。コード1（NumPy）とコード2（PyTorch）は各編の共通セクション A からコピー・ダウンロードできます。

### 多言語版

- 既定の言語は韓国語です。`en/` `zh/` `ja/` `uz/` `ru/` フォルダーに**英語・中国語・日本語・ウズベク語・ロシア語**の完全な翻訳版があります。本文、ボタン、用語の吹き出し、クイズ、Colab コードのコメントや出力メッセージまで翻訳されています。
- **自動切り替え：**韓国語のページを初めて開くと、ブラウザーの言語を見て対応する言語の同じページへ自動で移動します。対応していない言語の場合は韓国語のまま表示します。
- **言語の選択：**すべてのページの**最上部メニュー**に 🌐 言語メニューがあり、いちばん上の**言語バナー**からも切り替えられます。選んだ言語はブラウザーが記憶します。バナーの × は今回の閲覧中だけバナーを隠します。
- リンクの末尾に `?lang=stay` を付けると、自動切り替えなしでそのページを開きます。

### 見方

- **GitHub Pages：**Settings → Pages → Branch `main`、フォルダー `/ (root)` → Save。1〜3分後に `https://<ユーザー名>.github.io/<リポジトリ名>/` でホームが開きます。
- **自分のパソコン：**HTML ファイルをブラウザーで開きます。図とコードはすべてファイルに埋め込まれているので、1つのファイルだけでも表示できます。編や言語を行き来するにはフォルダー構成をそのまま保ってください。
- **Colab：**各編の共通セクション A の「ノートブック(.ipynb)をダウンロード」を押し、Colab で ファイル → ノートブックをアップロード で開きます。

外部から読み込むのはフォント（Google Fonts の IBM Plex Sans KR、jsDelivr の Galmuri）だけです。オフラインではシステムフォントに置き換わり、それ以外はそのまま動作します。

### ライセンス

この著作物は[クリエイティブ・コモンズ 表示-非営利 4.0 国際ライセンス（CC BY-NC 4.0）](https://creativecommons.org/licenses/by-nc/4.0/deed.ja)で公開しています。非営利目的に限り自由に共有・改変できますが、出典の表示が必要です。詳細と第三者の素材の表記は [LICENSE.md](LICENSE.md) にあります。

出典表記の例：Sangsun Ahn「ディープラーニング A to Z · ショウジョウバエ特別編」AI Model Review。

---

<a id="ozbekcha"></a>

## Oʻzbekcha

**Chuqur oʻrganish A dan Z gacha · Meva pashshasi maxsus soni** — meva pashshasi miyasi boʻyicha tadqiqotlarni sunʼiy neyron tarmoqlar bilan yonma-yon qoʻyib, tugmalarni bosib rasm va sonlarni oʻzgartirgan holda oʻrganiladigan uchta taʼlimiy veb-sahifa.

Muallif · Sangsun Ahn (M-Robo MChJ rahbari)

### Qismlar roʻyxati

| Qism | Fayl | Mazmuni |
|---|---|---|
| Bosh sahifa | [uz/index.html](uz/index.html) | Laboratoriya jamoasi, tadqiqot xaritasi, foydalanish tartibi |
| Maxsus 1 | [uz/1_fruitfly_brain_deep_learning.html](uz/1_fruitfly_brain_deep_learning.html) | Meva pashshasi miyasini oʻrganish — sezgidan anglashgacha, neyron tarmoq qadam-baqadam kattalashadi |
| Maxsus 2 | [uz/2_fruitfly_memory_retrieval_paper.html](uz/2_fruitfly_memory_retrieval_paper.html) | Oʻrganilgan xotira instinkt zanjiri orqali chaqirib olinadi — Dolan va boshq. (2018) *Neuron* maqolasining tasviriy talqini |
| Maxsus 3 | [uz/3_fruitfly_minecraft_connectome.html](uz/3_fruitfly_minecraft_connectome.html) | Minecraft ichidagi meva pashshasi miyasi — faqat ulanish sxemasi bilan ishlaydigan nerv tizimi |
| Oʻqituvchilar uchun | [uz/answers.html](uz/answers.html) | Nazorat savollari javoblari, baholash mezonlari, Colab topshiriqlari uchun namunaviy yechimlar |

Har bir qismda A–F umumiy boʻlimlari (ikkita Colab kodi, AI uchun promptlar, atamalar lugʻati, oʻxshatishlar, muallif haqida va adabiyotlar, oʻqituvchi uchun TIP) hamda yakuniy sahna bor. Kod 1 (NumPy) va Kod 2 (PyTorch) har bir qismning A umumiy boʻlimidan nusxalanadi yoki yuklab olinadi.

### Koʻp tilli versiyalar

- Asosiy til — koreys tili. `en/` `zh/` `ja/` `uz/` `ru/` papkalarida **ingliz, xitoy, yapon, oʻzbek va rus** tillaridagi toʻliq tarjimalar joylashgan: matn, tugmalar, atama izohlari, viktorinalar, Colab kodidagi izohlar va chiqish xabarlari ham tarjima qilingan.
- **Avtomatik almashish:** koreyscha sahifa birinchi marta ochilganda brauzer tili tekshiriladi va qoʻllab-quvvatlanadigan tildagi xuddi shu sahifaga oʻtiladi. Aks holda koreyscha sahifa qoladi.
- **Tilni tanlash:** har bir sahifaning **eng yuqori menyusida** 🌐 til menyusi, eng tepasida esa **til banneri** bor. Tanlangan til brauzerda eslab qolinadi. Banner ustidagi × uni faqat joriy tashrif davomida yashiradi.
- Havola oxiriga `?lang=stay` qoʻshilsa, sahifa avtomatik almashishsiz ochiladi.

### Koʻrish usullari

- **GitHub Pages:** Settings → Pages → Branch `main`, papka `/ (root)` → Save. 1–3 daqiqadan keyin bosh sahifa `https://<foydalanuvchi>.github.io/<repozitoriy>/` manzilida ochiladi.
- **Kompyuteringizda:** HTML faylni brauzerda oching. Barcha rasm va kodlar fayl ichiga joylangan, shuning uchun bitta faylning oʻzi ham ishlaydi. Qismlar va tillar orasida oʻtish uchun papka tuzilmasini oʻzgartirmang.
- **Colab:** har bir qismning A umumiy boʻlimidagi “Daftar (.ipynb) yuklab olish” tugmasini bosing, soʻng Colab’da File → Upload notebook orqali oching.

Tashqaridan faqat shriftlar yuklanadi (Google Fonts’dan IBM Plex Sans KR, jsDelivr’dan Galmuri). Oflayn rejimda tizim shriftlari ishlatiladi, qolgan hammasi odatdagidek ishlaydi.

### Litsenziya

Ushbu asar [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.en) litsenziyasi asosida eʼlon qilingan. Uni faqat notijorat maqsadlarda erkin tarqatish va oʻzgartirish mumkin, manba albatta koʻrsatilishi kerak. Batafsil maʼlumot va uchinchi tomon materiallari [LICENSE.md](LICENSE.md) faylida.

Manba koʻrsatish namunasi: Sangsun Ahn, “Chuqur oʻrganish A dan Z gacha · Meva pashshasi maxsus soni”, AI Model Review.

---

<a id="ru"></a>

## Русский

**Глубокое обучение от A до Z · Специальный выпуск о дрозофиле** — три учебные веб-страницы, на которых исследования мозга дрозофилы сопоставляются с искусственными нейронными сетями. Учиться предлагается, нажимая кнопки и меняя рисунки и числа.

Автор · Sangsun Ahn (генеральный директор M-Robo Inc.)

### Содержание

| Выпуск | Файл | О чём |
|---|---|---|
| Главная | [ru/index.html](ru/index.html) | Обитатели лаборатории, карта исследования, как пользоваться |
| Выпуск 1 | [ru/1_fruitfly_brain_deep_learning.html](ru/1_fruitfly_brain_deep_learning.html) | Путешествие по мозгу дрозофилы — от ощущения к распознаванию; нейросеть растёт шаг за шагом |
| Выпуск 2 | [ru/2_fruitfly_memory_retrieval_paper.html](ru/2_fruitfly_memory_retrieval_paper.html) | Выученное воспоминание извлекается через цепь инстинкта — иллюстрированный разбор статьи Dolan и др. (2018), *Neuron* |
| Выпуск 3 | [ru/3_fruitfly_minecraft_connectome.html](ru/3_fruitfly_minecraft_connectome.html) | Мозг дрозофилы в Minecraft — нервная система, которая работает только по схеме соединений |
| Для преподавателя | [ru/answers.html](ru/answers.html) | Ответы на контрольные вопросы, критерии оценивания, примеры решений заданий в Colab |

В каждом выпуске есть общие разделы A–F (два кода для Colab, промпты для ИИ, глоссарий, аналогии, об авторе и литература, советы преподавателю) и заключительная сцена. Код 1 (NumPy) и Код 2 (PyTorch) можно скопировать или скачать в общем разделе A каждого выпуска.

### Языковые версии

- Язык по умолчанию — корейский. В папках `en/` `zh/` `ja/` `uz/` `ru/` лежат полные переводы на **английский, китайский, японский, узбекский и русский**: текст, кнопки, всплывающие подсказки к терминам, тесты, комментарии и выводимые сообщения в коде Colab.
- **Автоматическое переключение:** при первом открытии корейской страницы определяется язык браузера, и если он поддерживается, открывается та же страница на этом языке. Иначе остаётся корейская версия.
- **Выбор языка:** на каждой странице есть меню 🌐 в **верхней панели навигации** и **языковой баннер** в самом верху. Выбранный язык запоминается браузером. Кнопка × скрывает баннер только на время текущего посещения.
- Добавьте `?lang=stay` к ссылке, чтобы открыть страницу без автоматического переключения.

### Как смотреть

- **GitHub Pages:** Settings → Pages → Branch `main`, папка `/ (root)` → Save. Через 1–3 минуты главная страница откроется по адресу `https://<имя>.github.io/<репозиторий>/`.
- **На своём компьютере:** откройте HTML-файл в браузере. Все рисунки и код встроены в файл, поэтому достаточно и одного файла. Чтобы переходить между выпусками и языками, сохраняйте структуру папок.
- **Colab:** нажмите «Скачать блокнот (.ipynb)» в общем разделе A, затем в Colab выберите Файл → Загрузить блокнот.

Извне загружаются только шрифты (IBM Plex Sans KR из Google Fonts, Galmuri из jsDelivr). Без интернета используются системные шрифты, всё остальное работает как обычно.

### Лицензия

Произведение распространяется по [лицензии Creative Commons «Атрибуция — Некоммерческое использование» 4.0 Международная (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.ru). Его можно свободно распространять и изменять в некоммерческих целях с обязательным указанием источника. Подробности и сведения о материалах третьих лиц — в [LICENSE.md](LICENSE.md).

Пример указания источника: Sangsun Ahn, «Глубокое обучение от A до Z · Специальный выпуск о дрозофиле», AI Model Review.
