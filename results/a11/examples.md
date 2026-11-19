# E4：最终评估器与 Gold 顺序相反的系统对（供人工挑选正文例子）

每条：同一文档中 Gold 认为 A 比 B 好、评估器却给 B 更高分；列出 A、B 不同的空隙（⟂ 为空隙，⏎ 为段落换行）。

## gum（场景：nl）

**1. GUM_textbook_artwork**（textbook）：Gold 更好 wtp F1 0.875 vs punkt 0.794；分数 0.8403 vs 0.8679；不同空隙 15 个
- `…ng the 24 aesthetic categories of poetry listed by Sikong Tu⟂ 司空图 (837–908) in his celebrated treatise (Sikong n.d.), onl…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…n of beauty. Moreover, according to Xu Hong’s (c. 1580–1660)⟂ Xishan Qinkuang 谿山琴况 (c. 1640), the most influential treati…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…eover, according to Xu Hong’s (c. 1580–1660) Xishan Qinkuang⟂ 谿山琴况 (c. 1640), the most influential treatise on the aesthe…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…tranquillity, clarity, profundity, antiquity, subtlety. [38]⟂ In Chinese paintings, prominence was given to the categorie…` — Gold 切，p = 0.968，左侧 terminal，wtp 切
- `… even less productive than in the contemporary artworld. 38.⟂ 和, 静, 清, 远, 古, 澹 (Research Institute of Music 2010, vol.14,…` — Gold 不切，p = 0.758，左侧 terminal，punkt 切
- `…2010, vol.14, 316–321). CONTEMPORARY DISCOURSE: BRIEF REMARK⟂ The Hegelian definition of aesthetics as the philosophy of …` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…n updated to “what makes an artwork meaningful or valuable?”⟂ Two factors might contribute to this situation. Firstly, it…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `… the involvement of an “internal sense” in the subject. [39]⟂ Hume argues that “beauty is no quality in things themselves…` — Gold 切，p = 0.936，左侧 terminal，wtp 切

**2. GUM_fiction_sneeze**（fiction）：Gold 更好 stanza F1 0.958 vs punkt 0.88；分数 0.947 vs 0.9495；不同空隙 22 个
- `… “I think I’ve got a fairy stuck up my nose.” “S’ not true!”⟂ came a voice from inside the nose in question. “Hey! Not ba…` — Gold 不切，p = 0.674，左侧 terminal，punkt 切
- `…getting worse. I pulled out a paper tissue. “Stop that now!”⟂ came the voice again. “You know I don’t like it.” “S’ not m…` — Gold 不切，p = 0.396，左侧 terminal，punkt 切
- `… pretty gross …” he said. “Got any other brilliant remarks?”⟂ I said. I held a finger up to my lips, and waved a hand to …` — Gold 不切，p = 0.602，左侧 terminal，punkt 切
- `…waved a hand to silence Pete’s reply. “Are you still there?”⟂ I called. “No thanks to you!” came the little voice. Pete l…` — Gold 不切，p = 0.515，左侧 terminal，punkt 切
- `… reply. “Are you still there?” I called. “No thanks to you!”⟂ came the little voice. Pete looked at me sort of sideways. …` — Gold 不切，p = 0.271，左侧 terminal，punkt 切
- `…at me sort of sideways. “Look, the first time was OK, but —”⟂ “It’s not a joke. I’ve really got something stuck up there.…` — Gold 切，p = 0.969，左侧 no punct.，段落边界，stanza 切
- `…then it started insulting me.” “You’re having me on, right?”⟂ “Pete! Look at me. If it’s a trick, how do I do it then?” P…` — Gold 切，p = 0.999，左侧 terminal，段落边界，punkt 切
- `…where it is if you catch them?” “That’s leprechauns, idiot!”⟂ said the voice from my noise. “So what’s the difference? Ju…` — Gold 不切，p = 0.474，左侧 terminal，punkt 切

**3. GUM_academic_mutation**（academic）：Gold 更好 sat F1 0.925 vs punkt 0.851；分数 0.8859 vs 0.8888；不同空隙 12 个
- `…utation Operators for Testing Geographic Information Systems⟂ Suilen H. Alvarado Laboratorio de Bases de Datos Campus de …` — Gold 切，p = 0.996，左侧 no punct.，段落边界，sat 切
- `…rsidade da Coruña, 15071 A Coruña, Spain; s.hernandez@udc.es⟂ Presented at the 2nd XoveTIC Congress, A Coruña, Spain, 5–6…` — Gold 切，p = 0.038，左侧 no punct.，sat 切
- `…TIC Congress, A Coruña, Spain, 5–6 September 2019. Abstract:⟂ In this article, we propose the definition of specific muta…` — Gold 切，p = 0.009，左侧 comma/colon/dash，sat 切
- `…o two real-world applications. Keywords: mutation operators;⟂ geographic information systems; mutation testing 1. Introdu…` — Gold 不切，p = 0.006，左侧 comma/colon/dash，sat 切
- `… operators; geographic information systems; mutation testing⟂ 1. Introduction Mutation-based testing [1] is a test techni…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，sat 切
- `…raphic information systems; mutation testing 1. Introduction⟂ Mutation-based testing [1] is a test technique that involve…` — Gold 切，p = 0.997，左侧 no punct.，段落边界，sat 切
- `…s and we generate the mutants. 2. Mutation Operators for GIS⟂ As a previous step to designing the mutation operators, we …` — Gold 切，p = 1.0，左侧 no punct.，段落边界，sat 切
- `…world GIS applications. ChangeCoordSys Operator (Listing 1):⟂ It exchanges the coordinate system of a geometry, so it doe…` — Gold 切，p = 0.02，左侧 comma/colon/dash，sat 切

**4. GUM_academic_epistemic**（academic）：Gold 更好 sat F1 0.986 vs punkt 0.914；分数 0.9677 vs 0.9854；不同空隙 5 个
- `…Epistemic pollution⟂ Agents can rationally choose between experts only if the cr…` — Gold 切，p = 0.983，左侧 no punct.，段落边界，sat 切
- `…the fact that authors are expected to pay a publication fee.⟂ 8 Yet some Frontiers journals appear to have engaged in bad…` — Gold 不切，p = 0.563，左侧 terminal，punkt 切
- `…e fact that authors are expected to pay a publication fee. 8⟂ Yet some Frontiers journals appear to have engaged in bad b…` — Gold 切，p = 0.209，左侧 no punct.，sat 切
- `…ction but also from bad behavior by legitimate institutions.⟂ 9 For example, pollution may result from attempts to game t…` — Gold 不切，p = 0.984，左侧 terminal，punkt 切
- `…ion but also from bad behavior by legitimate institutions. 9⟂ For example, pollution may result from attempts to game the…` — Gold 切，p = 0.228，左侧 no punct.，sat 切

**5. GUM_news_election**（news）：Gold 更好 stanza F1 0.901 vs punkt 0.847；分数 0.8514 vs 0.8992；不同空隙 8 个
- `… in Hong Kong legislative election Friday, September 9, 2016⟂ The Hong Kong legislative election was held on Sunday, with…` — Gold 切，p = 1.0，左侧 no punct.，段落边界，stanza 切
- `…establishment and pro-Beijing campus. Election Results Eddie⟂ Chu received the largest vote in any geographical constitue…` — Gold 不切，p = 0.0，左侧 no punct.，stanza 切
- `…u received the largest vote in any geographical constituency⟂ In the New Territories West geographical constituency, nonp…` — Gold 切，p = 0.473，左侧 no punct.，stanza 切
- `…ive thousand votes. Michael Tien of NPP DAB's Elizabeth Quat⟂ In the New Territories East geographical constituency, pan-…` — Gold 切，p = 0.02，左侧 no punct.，stanza 切
- `…eat. Alvin Yeung of Civic Party New People Party's Regina Ip⟂ In the Hong Kong Island geographical constituency, pro-Beij…` — Gold 切，p = 0.046，左侧 no punct.，stanza 切
- `…this constituency. Nathan Law of Demosisto Ann Chiang of DAB⟂ In the Kowloon West geographical constituency, pro-Beijing …` — Gold 切，p = 0.059，左侧 no punct.，stanza 切
- `…by 424 votes. Claudia Mo of Civic Party Wong Kwok-kin of FTU⟂ In the Kowloon East geographical constituency, pro-Beijing …` — Gold 切，p = 0.013，左侧 no punct.，stanza 切
- `…g Kwok-hing of FTU was defeated by 10,694 votes by James To.⟂ DAB's Starry Lee; from file, 2015. In traditional functiona…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切

**6. GUM_bio_galois**（bio）：Gold 更好 wtp F1 0.653 vs punkt 0.605；分数 0.7032 vs 0.76；不同空隙 30 个
- `…Final days Siméon Poisson reviewed⟂ Galois' paper on equation theory and declared it "incompreh…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…paper on equation theory and declared it "incomprehensible".⟂ Galois returned to mathematics after his expulsion from the…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `…, as it seemed that his political activism had priority. [4]⟂ [7] Siméon Poisson asked him to submit his work on the theo…` — Gold 不切，p = 0.003，左侧 terminal，wtp 切
- `… it seemed that his political activism had priority. [4] [7]⟂ Siméon Poisson asked him to submit his work on the theory o…` — Gold 切，p = 0.428，左侧 terminal，wtp 切
- `…; however, the rejection report ends on an encouraging note:⟂ "We would then suggest that the author should publish the w…` — Gold 不切，p = 0.006，左侧 comma/colon/dash，wtp 切
- `…ole of his work in order to form a definitive opinion." [16]⟂ While Poisson's report was made before Galois' July 14 arre…` — Gold 切，p = 0.784，左侧 terminal，wtp 切
- `…pinion." [16] While Poisson's report was made before Galois'⟂ July 14 arrest, it took until October to reach Galois in pr…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…nued polishing his ideas until his release on 29 April 1832,⟂ [12] after which he was somehow talked into a duel. [8] Gal…` — Gold 不切，p = 0.0，左侧 comma/colon/dash，wtp 切

**7. GUM_textbook_artwork**（textbook）：Gold 更好 stanza F1 0.984 vs sat 0.938；分数 0.9651 vs 1.0；不同空隙 3 个
- `… even less productive than in the contemporary artworld. 38.⟂ 和, 静, 清, 远, 古, 澹 (Research Institute of Music 2010, vol.14,…` — Gold 不切，p = 0.758，左侧 terminal，sat 切
- `…in which there is Uniformity amidst Variety” (§ VI, 67). 40.⟂ “Beauty is no quality in things themselves: it exists merel…` — Gold 不切，p = 0.832，左侧 terminal，sat 切
- `…ain the real sweet or real bitter” (Hume 1998, 136–137). 41.⟂ Denis Dutton (2010) provides such an account in his TED tal…` — Gold 不切，p = 0.976，左侧 terminal，sat 切

**8. GUM_fiction_rose**（fiction）：Gold 更好 sat F1 0.955 vs punkt 0.911；分数 0.9787 vs 0.9882；不同空隙 4 个
- `…Stranger Things Happen - Water Off a Black Dog’s Back⟂ Rachel Rook took Carroll home to meet her parents two month…` — Gold 切，p = 1.0，左侧 no punct.，段落边界，sat 切
- `…d driveby tourists. Do you think your parents will like me?"⟂ he said. He had spent the afternoon preparing for this visi…` — Gold 不切，p = 0.782，左侧 terminal，punkt 切
- `…ound it difficult to look directly at her. "Can I help you?"⟂ he said. She placed a shredded rectangle on the desk, and C…` — Gold 不切，p = 0.735，左侧 terminal，punkt 切
- `…maged book," she said. "What happened? Did your dog eat it?"⟂ he said, making a joke. "Yes," she said, and smiled. "What'…` — Gold 不切，p = 0.842，左侧 terminal，punkt 切

**9. GUM_textbook_history**（textbook）：Gold 更好 wtp F1 0.84 vs punkt 0.797；分数 0.8276 vs 0.8305；不同空隙 26 个
- `…1⟂ The Americas, Europe, and Africa Before 1492 1.1 The Americ…` — Gold 不切，p = 0.15，左侧 no punct.，wtp 切
- `…The Americas Learning Objectives By the end of this section,⟂ you will be able to: Locate on a map the major American civ…` — Gold 不切，p = 0.001，左侧 comma/colon/dash，wtp 切
- `…this section, you will be able to: Locate on a map the major⟂ American civilizations before the arrival of the Spanish Di…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…2 (credit: modification of work by Architect of the Capitol)⟂ Some scholars believe that between nine and fifteen thousan…` — Gold 切，p = 0.77，左侧 no punct.，wtp 切
- `…is more obvious than in Mesoamerica (Figure 1.3). Figure 1.3⟂ This map shows the extent of the major civilizations of the…` — Gold 切，p = 0.162，左侧 no punct.，wtp 切
- `…the interior of the continent less favorable for settlement.⟂ THE FIRST AMERICANS: THE OLMEC Mesoamerica is the geographi…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `…ontinent less favorable for settlement. THE FIRST AMERICANS:⟂ THE OLMEC Mesoamerica is the geographic area stretching fro…` — Gold 不切，p = 0.061，左侧 comma/colon/dash，wtp 切
- `…d the only known written language in the Western Hemisphere;⟂ researchers have made much progress in interpreting the ins…` — Gold 不切，p = 0.143，左侧 comma/colon/dash，wtp 切

**10. GUM_bio_jerome**（bio）：Gold 更好 wtp F1 0.598 vs punkt 0.556；分数 0.6582 vs 0.6918；不同空隙 31 个
- `…Jerome Jerome (/ dʒəˈroʊm /;⟂ Latin: Eusebius Sophronius Hieronymus; Greek: Εὐσέβιος Σωφρ…` — Gold 不切，p = 0.014，左侧 comma/colon/dash，wtp 切
- `…Jerome Jerome (/ dʒəˈroʊm /; Latin:⟂ Eusebius Sophronius Hieronymus; Greek: Εὐσέβιος Σωφρόνιος Ἱ…` — Gold 不切，p = 0.002，左侧 comma/colon/dash，wtp 切
- `…Jerome Jerome (/ dʒəˈroʊm /; Latin: Eusebius⟂ Sophronius Hieronymus; Greek: Εὐσέβιος Σωφρόνιος Ἱερώνυμος;…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…Jerome (/ dʒəˈroʊm /; Latin: Eusebius Sophronius Hieronymus;⟂ Greek: Εὐσέβιος Σωφρόνιος Ἱερώνυμος; c. 347 – 30 September …` — Gold 不切，p = 0.015，左侧 comma/colon/dash，wtp 切
- `…(/ dʒəˈroʊm /; Latin: Eusebius Sophronius Hieronymus; Greek:⟂ Εὐσέβιος Σωφρόνιος Ἱερώνυμος; c. 347 – 30 September 420) wa…` — Gold 不切，p = 0.003，左侧 comma/colon/dash，wtp 切
- `…ʊm /; Latin: Eusebius Sophronius Hieronymus; Greek: Εὐσέβιος⟂ Σωφρόνιος Ἱερώνυμος; c. 347 – 30 September 420) was a Latin…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `… Sophronius Hieronymus; Greek: Εὐσέβιος Σωφρόνιος Ἱερώνυμος;⟂ c. 347 – 30 September 420) was a Latin Catholic priest, con…` — Gold 不切，p = 0.01，左侧 comma/colon/dash，wtp 切
- `…llage near Emona on the border of Dalmatia and Pannonia. [3]⟂ [4] [5] He is best known for his translation of most of the…` — Gold 不切，p = 0.001，左侧 terminal，wtp 切

**11. GUM_academic_epistemic**（academic）：Gold 更好 sat F1 0.986 vs stanza 0.944；分数 0.9677 vs 0.9703；不同空隙 3 个
- `…Epistemic pollution⟂ Agents can rationally choose between experts only if the cr…` — Gold 切，p = 0.983，左侧 no punct.，段落边界，sat 切
- `…the fact that authors are expected to pay a publication fee.⟂ 8 Yet some Frontiers journals appear to have engaged in bad…` — Gold 不切，p = 0.563，左侧 terminal，stanza 切
- `…ction but also from bad behavior by legitimate institutions.⟂ 9 For example, pollution may result from attempts to game t…` — Gold 不切，p = 0.984，左侧 terminal，stanza 切

**12. GUM_textbook_alamo**（textbook）：Gold 更好 stanza F1 0.97 vs punkt 0.938；分数 0.9513 vs 0.9673；不同空隙 3 个
- `…rd Mexicans as dishonest, ignorant, and backward. Figure 1.3⟂ This 1833 map shows the extent of land grants made by Mexic…` — Gold 切，p = 0.225，左侧 no punct.，stanza 切
- `…egislature, and removed restrictions on commerce. Figure 1.4⟂ This portrait of General Antonio Lopez de Santa Anna depict…` — Gold 切，p = 0.243，左侧 no punct.，stanza 切
- `… freed by the Mexican Army. Terrified, they fled. Figure 1.5⟂ The Fall of the Alamo, painted by Theodore Gentilz fewer th…` — Gold 切，p = 0.198，左侧 no punct.，stanza 切

**13. GUM_fiction_rose**（fiction）：Gold 更好 stanza F1 0.943 vs punkt 0.911；分数 0.9663 vs 0.9882；不同空隙 3 个
- `…d driveby tourists. Do you think your parents will like me?"⟂ he said. He had spent the afternoon preparing for this visi…` — Gold 不切，p = 0.782，左侧 terminal，punkt 切
- `…ound it difficult to look directly at her. "Can I help you?"⟂ he said. She placed a shredded rectangle on the desk, and C…` — Gold 不切，p = 0.735，左侧 terminal，punkt 切
- `…maged book," she said. "What happened? Did your dog eat it?"⟂ he said, making a joke. "Yes," she said, and smiled. "What'…` — Gold 不切，p = 0.842，左侧 terminal，punkt 切

**14. GUM_fiction_veronique**（fiction）：Gold 更好 sat F1 0.992 vs punkt 0.961；分数 0.9787 vs 0.9966；不同空隙 4 个
- `…The Cost to Be Wise⟂ Veronique stayed with me that night, lying next to me in my…` — Gold 切，p = 0.409，左侧 no punct.，sat 切
- `… She lay tense in the dark. "Are you going to be a teacher?"⟂ I asked. She laughed. "If I get out of here." I waited for …` — Gold 不切，p = 0.803，左侧 terminal，punkt 切
- `…e she meant if she made herself. "You come here from Earth?"⟂ I asked. To get her to talk, although I was tired of lingua…` — Gold 不切，p = 0.82，左侧 terminal，punkt 切
- `…ed her talking. "History, do you know the word ' history '?"⟂ she asked. Of course I knew the word "history." "I study hi…` — Gold 不切，p = 0.789，左侧 terminal，punkt 切

**15. GUM_academic_epistemic**（academic）：Gold 更好 stanza F1 0.944 vs punkt 0.914；分数 0.9703 vs 0.9854；不同空隙 2 个
- `…e fact that authors are expected to pay a publication fee. 8⟂ Yet some Frontiers journals appear to have engaged in bad b…` — Gold 切，p = 0.209，左侧 no punct.，stanza 切
- `…ion but also from bad behavior by legitimate institutions. 9⟂ For example, pollution may result from attempts to game the…` — Gold 切，p = 0.228，左侧 no punct.，stanza 切

**16. GUM_fiction_frankenstein**（fiction）：Gold 更好 sat F1 0.983 vs punkt 0.961；分数 0.9931 vs 0.9956；不同空隙 4 个
- `…se,” he says. “Where’s your mom? Is your mom here?” “Sorry!”⟂ I say. The chubby guy frowns at me. “Cara,” I say. “No hand…` — Gold 不切，p = 0.423，左侧 terminal，punkt 切
- `…at me, looks at the lights. I gently try to take them. “No!”⟂ she wails. “Pretty!” “I’m sorry,” I say, “I’m her brother. …` — Gold 不切，p = 0.456，左侧 terminal，punkt 切
- `… and she follows me, gulping and wailing. “What did you do?”⟂ my mom says when she sees us. My mom had to buy stuff, like…` — Gold 不切，p = 0.757，左侧 terminal，punkt 切
- `…’s got that white look around her mouth. “Robert,” Mom says,⟂ “I’m going to have to take her to the Emergency Room.” “Oka…` — Gold 切，p = 0.465，左侧 comma/colon/dash，段落边界，sat 切

**17. GUM_bio_goode**（bio）：Gold 更好 wtp F1 0.786 vs punkt 0.766；分数 0.8 vs 0.8193；不同空隙 11 个
- `… a lead role in the critically acclaimed British mini-serial⟂ Dancing on the Edge, as music journalist Stanley Mitchell, …` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…nomination for Outstanding Guest Star in a Drama Series. [2]⟂ He currently stars in Sky One's fantasy-romance series, A D…` — Gold 切，p = 0.799，左侧 terminal，wtp 切
- `…hew Vaughn's Kingsman prequel, The King's Man. Personal life⟂ Goode has been in a relationship with Sophie Dymoke since 2…` — Gold 切，p = 0.995，左侧 no punct.，段落边界，wtp 切
- `…ip with Sophie Dymoke since 2005. They married in 2014. [17]⟂ [18] [19] They have two daughters, Matilda Eve Goode [3] (b…` — Gold 不切，p = 0.001，左侧 terminal，wtp 切
- `…th Sophie Dymoke since 2005. They married in 2014. [17] [18]⟂ [19] They have two daughters, Matilda Eve Goode [3] (b. Mar…` — Gold 不切，p = 0.001，左侧 terminal，wtp 切
- `…phie Dymoke since 2005. They married in 2014. [17] [18] [19]⟂ They have two daughters, Matilda Eve Goode [3] (b. March 20…` — Gold 切，p = 0.662，左侧 terminal，wtp 切
- `…17] [18] [19] They have two daughters, Matilda Eve Goode [3]⟂ (b. March 2009) and Teddie Eleanor Rose Goode (b. September…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…[18] [19] They have two daughters, Matilda Eve Goode [3] (b.⟂ March 2009) and Teddie Eleanor Rose Goode (b. September 201…` — Gold 不切，p = 0.027，左侧 terminal，punkt 切

**18. GUM_bio_theodorus**（bio）：Gold 更好 stanza F1 0.947 vs punkt 0.931；分数 0.9466 vs 0.947；不同空隙 1 个
- `… as the head of the communities in both title and authority.⟂ [1]…` — Gold 不切，p = 0.478，左侧 terminal，punkt 切

**19. GUM_fiction_frankenstein**（fiction）：Gold 更好 stanza F1 0.977 vs punkt 0.961；分数 0.976 vs 0.9956；不同空隙 7 个
- `…touch those,” he says. “Where’s your mom? Is your mom here?”⟂ “Sorry!” I say. The chubby guy frowns at me. “Cara,” I say.…` — Gold 切，p = 0.998，左侧 terminal，段落边界，punkt 切
- `…se,” he says. “Where’s your mom? Is your mom here?” “Sorry!”⟂ I say. The chubby guy frowns at me. “Cara,” I say. “No hand…` — Gold 不切，p = 0.423，左侧 terminal，punkt 切
- `…at me, looks at the lights. I gently try to take them. “No!”⟂ she wails. “Pretty!” “I’m sorry,” I say, “I’m her brother. …` — Gold 不切，p = 0.456，左侧 terminal，punkt 切
- `…ights. I gently try to take them. “No!” she wails. “Pretty!”⟂ “I’m sorry,” I say, “I’m her brother. She’s developmentally…` — Gold 切，p = 0.997，左侧 terminal，段落边界，punkt 切
- `…ets me wipe her face off. “You ready to go back to see Mom?”⟂ I say. When we pass Spencer’s again, she stops. “Uhhh,” she…` — Gold 不切，p = 0.691，左侧 terminal，punkt 切
- `… and she follows me, gulping and wailing. “What did you do?”⟂ my mom says when she sees us. My mom had to buy stuff, like…` — Gold 不切，p = 0.757，左侧 terminal，punkt 切
- `…ll phone if you need to call.” “Can Toph and Len come over?”⟂ I ask. She sighs. “Okay. But no roughhousing. Remember you …` — Gold 不切，p = 0.671，左侧 terminal，punkt 切

**20. GUM_letter_conference**（letter）：Gold 更好 sat F1 1.0 vs stanza 0.985；分数 0.9845 vs 0.9915；不同空隙 1 个
- `…ne meeting thismorning, and found it cancelled. Best wishes,⟂ Bill…` — Gold 切，p = 0.255，左侧 comma/colon/dash，段落边界，sat 切

**21. GUM_court_mental**（court）：Gold 更好 stanza F1 0.972 vs punkt 0.958；分数 0.9727 vs 0.9852；不同空隙 1 个
- `…number DA22-0645 in the matter of the mental health of T. W.⟂ And I understand, uh, Miss Peterson, you're going to reserv…` — Gold 切，p = 0.051，左侧 terminal，stanza 切

**22. GUM_bio_hadid**（bio）：Gold 更好 stanza F1 0.947 vs sat 0.933；分数 0.9633 vs 0.97；不同空隙 4 个
- `…Zaha Hadid⟂ Dame Zaha Mohammad Hadid DBE RA (Arabic: زها حديد Zahā Ḥadī…` — Gold 切，p = 0.996，左侧 no punct.，段落边界，sat 切
- `…Zaha Hadid Dame Zaha Mohammad Hadid⟂ DBE RA (Arabic: زها حديد Zahā Ḥadīd; 31 October 1950 – 31 M…` — Gold 不切，p = 0.018，左侧 no punct.，sat 切
- `… for the 2022 FIFA World Cup. Early life and academic career⟂ Hadid was born on 31 October 1950 in Baghdad, Iraq, to an u…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，sat 切
- `…he joinery she knew we could fix that later. She was right.’⟂ Her fourth-year student project was a painting of a hotel i…` — Gold 切，p = 0.872，左侧 terminal，stanza 切

**23. GUM_bio_hadid**（bio）：Gold 更好 punkt F1 0.947 vs sat 0.933；分数 0.9633 vs 0.97；不同空隙 4 个
- `…Zaha Hadid⟂ Dame Zaha Mohammad Hadid DBE RA (Arabic: زها حديد Zahā Ḥadī…` — Gold 切，p = 0.996，左侧 no punct.，段落边界，sat 切
- `…Zaha Hadid Dame Zaha Mohammad Hadid⟂ DBE RA (Arabic: زها حديد Zahā Ḥadīd; 31 October 1950 – 31 M…` — Gold 不切，p = 0.018，左侧 no punct.，sat 切
- `… for the 2022 FIFA World Cup. Early life and academic career⟂ Hadid was born on 31 October 1950 in Baghdad, Iraq, to an u…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，sat 切
- `…he joinery she knew we could fix that later. She was right.’⟂ Her fourth-year student project was a painting of a hotel i…` — Gold 切，p = 0.872，左侧 terminal，punkt 切

**24. GUM_fiction_giants**（fiction）：Gold 更好 stanza F1 0.986 vs punkt 0.973；分数 0.9807 vs 0.9844；不同空隙 1 个
- `… of her face, who was incensed. “We live in the modern era!”⟂ she cried. “You should be ashamed of yourself!" The travele…` — Gold 不切，p = 0.604，左侧 terminal，punkt 切

**25. GUM_interview_peres**（interview）：Gold 更好 stanza F1 0.98 vs sat 0.968；分数 0.9831 vs 0.9832；不同空隙 6 个
- `…The future of the peace process in Israel⟂ "The future is only peace. The problem is how long will it …` — Gold 切，p = 0.975，左侧 no punct.，段落边界，sat 切
- `…it take and how many victims will it call for." Shimon Peres⟂ image: David Shankbone. Mr. President, as a Nobel Peace Pri…` — Gold 不切，p = 0.075，左侧 no punct.，sat 切
- `… and how many victims will it call for." Shimon Peres image:⟂ David Shankbone. Mr. President, as a Nobel Peace Prize reci…` — Gold 不切，p = 0.026，左侧 comma/colon/dash，sat 切
- `… you asked him to remember? The waning importance of history⟂ Isn’t the answer to that question that wise decisions are m…` — Gold 切，p = 0.991，左侧 no punct.，段落边界，sat 切
- `… us and they have all the oil in the world. We have brains."⟂ — Peres, on whether the Arab states' oil wealth will eclips…` — Gold 切，p = 0.215，左侧 terminal，stanza 切
- `…s and they have all the oil in the world. We have brains." —⟂ Peres, on whether the Arab states' oil wealth will eclipse …` — Gold 不切，p = 0.091，左侧 comma/colon/dash，sat 切

## gentle（场景：nl）

**1. GENTLE_poetry_annabel**（poetry）：Gold 更好 sat F1 0.97 vs punkt 0.455；分数 0.7241 vs 0.823；不同空隙 13 个
- `…en there lived whom you may know By the name of ANNABEL LEE;⟂ And this maiden she lived with no other thought Than to lov…` — Gold 切，p = 0.087，左侧 comma/colon/dash，sat 切
- `…was a child and she was a child, In this kingdom by the sea:⟂ But we loved with a love that was more than love — I and my…` — Gold 切，p = 0.411，左侧 comma/colon/dash，sat 切
- `… the sea: But we loved with a love that was more than love —⟂ I and my ANNABEL LEE; With a love that the winged seraphs o…` — Gold 切，p = 0.027，左侧 comma/colon/dash，sat 切
- `… with a love that was more than love — I and my ANNABEL LEE;⟂ With a love that the winged seraphs of heaven Coveted her a…` — Gold 切，p = 0.025，左侧 comma/colon/dash，sat 切
- `…wind blew out of a cloud, chilling My beautiful ANNABEL LEE;⟂ So that her highborn kinsman came And bore her away from me…` — Gold 切，p = 0.112，左侧 comma/colon/dash，sat 切
- `…gels, not half so happy in heaven, Went envying her and me —⟂ Yes! — that was the reason (as all men know, In this kingdo…` — Gold 切，p = 0.338，左侧 comma/colon/dash，sat 切
- `… not half so happy in heaven, Went envying her and me — Yes!⟂ — that was the reason (as all men know, In this kingdom by …` — Gold 不切，p = 0.028，左侧 terminal，punkt 切
- `…was the reason (as all men know, In this kingdom by the sea)⟂ That the wind came out of the cloud by night, Chilling and …` — Gold 不切，p = 0.003，左侧 no punct.，sat 切

**2. GENTLE_poetry_raven**（poetry）：Gold 更好 sat F1 0.823 vs punkt 0.634；分数 0.7183 vs 0.9207；不同空隙 49 个
- `…y, Over many a quaint and curious volume of forgotten lore —⟂ While I nodded, nearly napping, suddenly there came a tappi…` — Gold 切，p = 0.458，左侧 comma/colon/dash，sat 切
- `…Tis some visiter," I muttered, "tapping at my chamber door —⟂ Only this and nothing more." Ah, distinctly I remember it w…` — Gold 切，p = 0.058，左侧 comma/colon/dash，sat 切
- `…Eagerly I wished the morrow; — vainly I had sought to borrow⟂ From my books surcease of sorrow — sorrow for the lost Leno…` — Gold 不切，p = 0.001，左侧 no punct.，sat 切
- `…m my books surcease of sorrow — sorrow for the lost Lenore —⟂ For the rare and radiant maiden whom the angels name Lenore…` — Gold 不切，p = 0.017，左侧 comma/colon/dash，sat 切
- `…or the rare and radiant maiden whom the angels name Lenore —⟂ Nameless here for evermore. And the silken sad uncertain ru…` — Gold 不切，p = 0.058，左侧 comma/colon/dash，sat 切
- `…And the silken sad uncertain rustling of each purple curtain⟂ Thrilled me — filled me with fantastic terrors never felt b…` — Gold 不切，p = 0.003，左侧 no punct.，sat 切
- `…led me — filled me with fantastic terrors never felt before;⟂ So that now, to still the beating of my heart, I stood repe…` — Gold 切，p = 0.061，左侧 comma/colon/dash，sat 切
- `… "'Tis some visiter entreating entrance at my chamber door —⟂ Some late visiter entreating entrance at my chamber door; T…` — Gold 切，p = 0.024，左侧 comma/colon/dash，sat 切

**3. GENTLE_poetry_raven**（poetry）：Gold 更好 stanza F1 0.78 vs punkt 0.634；分数 0.702 vs 0.9207；不同空隙 44 个
- `…CE upon a midnight dreary, while I pondered, weak and weary,⟂ Over many a quaint and curious volume of forgotten lore — W…` — Gold 不切，p = 0.013，左侧 comma/colon/dash，stanza 切
- `…y, Over many a quaint and curious volume of forgotten lore —⟂ While I nodded, nearly napping, suddenly there came a tappi…` — Gold 切，p = 0.458，左侧 comma/colon/dash，stanza 切
- `…ile I nodded, nearly napping, suddenly there came a tapping,⟂ As of some one gently rapping, rapping at my chamber door. …` — Gold 不切，p = 0.005，左侧 comma/colon/dash，stanza 切
- `…Tis some visiter," I muttered, "tapping at my chamber door —⟂ Only this and nothing more." Ah, distinctly I remember it w…` — Gold 切，p = 0.058，左侧 comma/colon/dash，stanza 切
- `…re." Ah, distinctly I remember it was in the bleak December,⟂ And each separate dying ember wrought its ghost upon the fl…` — Gold 不切，p = 0.082，左侧 comma/colon/dash，stanza 切
- `…m my books surcease of sorrow — sorrow for the lost Lenore —⟂ For the rare and radiant maiden whom the angels name Lenore…` — Gold 不切，p = 0.017，左侧 comma/colon/dash，stanza 切
- `…led me — filled me with fantastic terrors never felt before;⟂ So that now, to still the beating of my heart, I stood repe…` — Gold 切，p = 0.061，左侧 comma/colon/dash，stanza 切
- `… "'Tis some visiter entreating entrance at my chamber door —⟂ Some late visiter entreating entrance at my chamber door; T…` — Gold 切，p = 0.024，左侧 comma/colon/dash，stanza 切

**4. GENTLE_poetry_annabel**（poetry）：Gold 更好 stanza F1 0.6 vs punkt 0.455；分数 0.4691 vs 0.823；不同空隙 18 个
- `…ANNABEL LEE. IT was many and many a year ago,⟂ In a kingdom by the sea, That a maiden there lived whom you…` — Gold 不切，p = 0.001，左侧 comma/colon/dash，stanza 切
- `…en there lived whom you may know By the name of ANNABEL LEE;⟂ And this maiden she lived with no other thought Than to lov…` — Gold 切，p = 0.087，左侧 comma/colon/dash，stanza 切
- `…was a child and she was a child, In this kingdom by the sea:⟂ But we loved with a love that was more than love — I and my…` — Gold 切，p = 0.411，左侧 comma/colon/dash，stanza 切
- `… Coveted her and me. And this was the reason that, long ago,⟂ In this kingdom by the sea, A wind blew out of a cloud, chi…` — Gold 不切，p = 0.001，左侧 comma/colon/dash，stanza 切
- `…wind blew out of a cloud, chilling My beautiful ANNABEL LEE;⟂ So that her highborn kinsman came And bore her away from me…` — Gold 切，p = 0.112，左侧 comma/colon/dash，stanza 切
- `… My beautiful ANNABEL LEE; So that her highborn kinsman came⟂ And bore her away from me, To shut her up in a sepulchre In…` — Gold 不切，p = 0.0，左侧 no punct.，stanza 切
- `…ame And bore her away from me, To shut her up in a sepulchre⟂ In this kingdom by the sea. The angels, not half so happy i…` — Gold 不切，p = 0.003，左侧 no punct.，stanza 切
- `…was the reason (as all men know, In this kingdom by the sea)⟂ That the wind came out of the cloud by night, Chilling and …` — Gold 不切，p = 0.003，左侧 no punct.，stanza 切

**5. GENTLE_poetry_flower**（poetry）：Gold 更好 stanza F1 0.8 vs punkt 0.667；分数 0.8015 vs 0.997；不同空隙 2 个
- `…I HIDE myself within my flower,⟂ That wearing on your breast, You, unsuspecting, wear me too…` — Gold 不切，p = 0.036，左侧 comma/colon/dash，stanza 切
- `…hat wearing on your breast, You, unsuspecting, wear me too —⟂ And angels know the rest. I hide myself within my flower, T…` — Gold 切，p = 0.392，左侧 comma/colon/dash，stanza 切

**6. GENTLE_poetry_flower**（poetry）：Gold 更好 sat F1 0.8 vs punkt 0.667；分数 0.785 vs 0.997；不同空隙 2 个
- `…hat wearing on your breast, You, unsuspecting, wear me too —⟂ And angels know the rest. I hide myself within my flower, T…` — Gold 切，p = 0.392，左侧 comma/colon/dash，sat 切
- `… That, fading from your vase, You, unsuspecting, feel for me⟂ Almost a loneliness.…` — Gold 不切，p = 0.006，左侧 no punct.，sat 切

**7. GENTLE_dictionary_trust**（dictionary）：Gold 更好 stanza F1 0.722 vs punkt 0.595；分数 0.7202 vs 0.7835；不同空隙 12 个
- `…: -ʌst Noun trust (countable and uncountable, plural trusts)⟂ Confidence in or reliance on some person or quality. He nee…` — Gold 切，p = 0.995，左侧 no punct.，段落边界，stanza 切
- `…om anything is confided; responsible charge or office. (law)⟂ The confidence vested in a person who has legal ownership o…` — Gold 切，p = 0.042，左侧 no punct.，stanza 切
- `…it of another. I put the house into my sister's trust. (law)⟂ An arrangement whereby property or money is given to be hel…` — Gold 切，p = 0.046，左侧 no punct.，stanza 切
- `…sting, simple past and past participle trusted) (transitive)⟂ To place confidence in, to rely on, to confide in. We canno…` — Gold 切，p = 0.016，左侧 no punct.，stanza 切
- `…cannot trust anyone who deceives us. (intransitive, with in)⟂ To have faith in; to rely on for continuing support or aid.…` — Gold 切，p = 0.032，左侧 no punct.，stanza 切
- `…ort or aid. In God We Trust (official US motto) (transitive)⟂ To give credence to; to believe; to credit. (transitive) To…` — Gold 切，p = 0.019，左侧 no punct.，stanza 切
- `…ve) To give credence to; to believe; to credit. (transitive)⟂ To hope confidently; to believe (usually with a phrase or i…` — Gold 切，p = 0.03，左侧 no punct.，stanza 切
- `… in a person by entrusting them with something. (transitive)⟂ To commit, as to one's care; to entrust. (transitive) To gi…` — Gold 切，p = 0.017，左侧 no punct.，stanza 切

**8. GENTLE_poetry_annabel**（poetry）：Gold 更好 wtp F1 0.538 vs punkt 0.455；分数 0.3107 vs 0.823；不同空隙 34 个
- `…ANNABEL LEE.⟂ IT was many and many a year ago, In a kingdom by the sea, T…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `…ANNABEL LEE. IT was many and many a year ago,⟂ In a kingdom by the sea, That a maiden there lived whom you…` — Gold 不切，p = 0.001，左侧 comma/colon/dash，wtp 切
- `…E. IT was many and many a year ago, In a kingdom by the sea,⟂ That a maiden there lived whom you may know By the name of …` — Gold 不切，p = 0.002，左侧 comma/colon/dash，wtp 切
- `…gdom by the sea, That a maiden there lived whom you may know⟂ By the name of ANNABEL LEE; And this maiden she lived with …` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…en there lived whom you may know By the name of ANNABEL LEE;⟂ And this maiden she lived with no other thought Than to lov…` — Gold 切，p = 0.087，左侧 comma/colon/dash，wtp 切
- `…ANNABEL LEE; And this maiden she lived with no other thought⟂ Than to love and be loved by me. I was a child and she was …` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…lived with no other thought Than to love and be loved by me.⟂ I was a child and she was a child, In this kingdom by the s…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `… love and be loved by me. I was a child and she was a child,⟂ In this kingdom by the sea: But we loved with a love that w…` — Gold 不切，p = 0.002，左侧 comma/colon/dash，wtp 切

**9. GENTLE_poetry_road**（poetry）：Gold 更好 sat F1 0.75 vs punkt 0.667；分数 0.7254 vs 0.7367；不同空隙 7 个
- `…diverged in a yellow wood, And sorry I could not travel both⟂ And be one traveler, long I stood And looked down one as fa…` — Gold 不切，p = 0.001，左侧 no punct.，sat 切
- `…n one as far as I could To where it bent in the undergrowth;⟂ Then took the other, as just as fair, And having perhaps th…` — Gold 切，p = 0.998，左侧 comma/colon/dash，段落边界，sat 切
- `…aps the better claim, Because it was grassy and wanted wear;⟂ Though as for that the passing there Had worn them really a…` — Gold 切，p = 0.092，左侧 comma/colon/dash，sat 切
- `… that the passing there Had worn them really about the same,⟂ And both that morning equally lay In leaves no step had tro…` — Gold 切，p = 0.984，左侧 comma/colon/dash，段落边界，sat 切
- `…I should ever come back. I shall be telling this with a sigh⟂ Somewhere ages and ages hence: Two roads diverged in a wood…` — Gold 不切，p = 0.004，左侧 no punct.，sat 切
- `…l be telling this with a sigh Somewhere ages and ages hence:⟂ Two roads diverged in a wood, and I — I took the one less t…` — Gold 不切，p = 0.174，左侧 comma/colon/dash，sat 切
- `…e ages and ages hence: Two roads diverged in a wood, and I —⟂ I took the one less travelled by, And that has made all the…` — Gold 不切，p = 0.041，左侧 comma/colon/dash，sat 切

**10. GENTLE_dictionary_school**（dictionary）：Gold 更好 wtp F1 0.71 vs stanza 0.648；分数 0.7565 vs 0.7779；不同空隙 40 个
- `…school English Pronunciation enPR:⟂ sko͞ol, IPA(key): / skuːl / Audio (Received Pronunciation) …` — Gold 不切，p = 0.01，左侧 comma/colon/dash，wtp 切
- `…school English Pronunciation enPR: sko͞ol,⟂ IPA(key): / skuːl / Audio (Received Pronunciation) Rhymes: …` — Gold 不切，p = 0.007，左侧 comma/colon/dash，wtp 切
- `…school English Pronunciation enPR: sko͞ol, IPA(key):⟂ / skuːl / Audio (Received Pronunciation) Rhymes: -uːl Etymo…` — Gold 不切，p = 0.022，左侧 comma/colon/dash，wtp 切
- `…hool English Pronunciation enPR: sko͞ol, IPA(key): / skuːl /⟂ Audio (Received Pronunciation) Rhymes: -uːl Etymology 1 Fro…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…(key): / skuːl / Audio (Received Pronunciation) Rhymes: -uːl⟂ Etymology 1 From Middle English scole, schole (“group of pe…` — Gold 切，p = 0.541，左侧 no punct.，wtp 切
- `…ːl / Audio (Received Pronunciation) Rhymes: -uːl Etymology 1⟂ From Middle English scole, schole (“group of persons, multi…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…itude, school of fish”). Doublet of shoal. Alternative forms⟂ skull (obsolete) Noun school (1) of fish. school (plural sc…` — Gold 切，p = 0.997，左侧 no punct.，段落边界，wtp 切
- `…). Doublet of shoal. Alternative forms skull (obsolete) Noun⟂ school (1) of fish. school (plural schools) (collective) A …` — Gold 切，p = 0.994，左侧 no punct.，段落边界，wtp 切

**11. GENTLE_poetry_raven**（poetry）：Gold 更好 wtp F1 0.678 vs punkt 0.634；分数 0.601 vs 0.9207；不同空隙 63 个
- `…THE RAVEN.⟂ ONCE upon a midnight dreary, while I pondered, weak and wea…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切
- `…CE upon a midnight dreary, while I pondered, weak and weary,⟂ Over many a quaint and curious volume of forgotten lore — W…` — Gold 不切，p = 0.013，左侧 comma/colon/dash，wtp 切
- `…y, Over many a quaint and curious volume of forgotten lore —⟂ While I nodded, nearly napping, suddenly there came a tappi…` — Gold 切，p = 0.458，左侧 comma/colon/dash，wtp 切
- `…ile I nodded, nearly napping, suddenly there came a tapping,⟂ As of some one gently rapping, rapping at my chamber door. …` — Gold 不切，p = 0.005，左侧 comma/colon/dash，wtp 切
- `…ly rapping, rapping at my chamber door. "'Tis some visiter,"⟂ I muttered, "tapping at my chamber door — Only this and not…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切
- `…re." Ah, distinctly I remember it was in the bleak December,⟂ And each separate dying ember wrought its ghost upon the fl…` — Gold 不切，p = 0.082，左侧 comma/colon/dash，wtp 切
- `…m my books surcease of sorrow — sorrow for the lost Lenore —⟂ For the rare and radiant maiden whom the angels name Lenore…` — Gold 不切，p = 0.017，左侧 comma/colon/dash，wtp 切
- `…And the silken sad uncertain rustling of each purple curtain⟂ Thrilled me — filled me with fantastic terrors never felt b…` — Gold 不切，p = 0.003，左侧 no punct.，wtp 切

**12. GENTLE_medical_anemia**（medical）：Gold 更好 punkt F1 0.842 vs stanza 0.818；分数 0.8596 vs 0.866；不同空隙 7 个
- `…anemia at the time of his CLL diagnosis. CHIEF COMPLAINT: 1.⟂ Chronic lymphocytic leukemia (CLL). 2. Autoimmune hemolytic…` — Gold 不切，p = 0.002，左侧 terminal，punkt 切
- `…. CHIEF COMPLAINT: 1. Chronic lymphocytic leukemia (CLL). 2.⟂ Autoimmune hemolytic anemia. 3. Oral ulcer. HISTORY OF PRES…` — Gold 不切，p = 0.001，左侧 terminal，punkt 切
- `…mphocytic leukemia (CLL). 2. Autoimmune hemolytic anemia. 3.⟂ Oral ulcer. HISTORY OF PRESENT ILLNESS: The patient is a 72…` — Gold 不切，p = 0.001，左侧 terminal，punkt 切
- `…w of systems is negative. PHYSICAL EXAM: VITALS: BP: 120/74.⟂ HEART RATE: 72. TEMP: 97.8. Weight: 112.7 kg. GEN: He has a…` — Gold 切，p = 0.925，左侧 terminal，punkt 切
- `…negative. PHYSICAL EXAM: VITALS: BP: 120/74. HEART RATE: 72.⟂ TEMP: 97.8. Weight: 112.7 kg. GEN: He has a obvious cold so…` — Gold 切，p = 0.178，左侧 terminal，punkt 切
- `…YSICAL EXAM: VITALS: BP: 120/74. HEART RATE: 72. TEMP: 97.8.⟂ Weight: 112.7 kg. GEN: He has a obvious cold sore on his bo…` — Gold 切，p = 0.596，左侧 terminal，punkt 切
- `…S: BP: 120/74. HEART RATE: 72. TEMP: 97.8. Weight: 112.7 kg.⟂ GEN: He has a obvious cold sore on his bottom lip on the le…` — Gold 切，p = 1.0，左侧 terminal，段落边界，punkt 切

**13. GENTLE_proof_wosets**（proof）：Gold 更好 stanza F1 0.869 vs wtp 0.846；分数 0.855 vs 0.8757；不同空隙 30 个
- `…Proof⟂ If the sets S and T considered are empty or singletons, the…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，stanza 切
- `… S and T each contain at least two elements. Let U = (S, ⪯S)⟂ ∪ (T, ⪯T) Define the following relation ⪯ on U: ∀ x, y ∈ U:…` — Gold 不切，p = 0.007，左侧 no punct.，wtp 切
- `… and T each contain at least two elements. Let U = (S, ⪯S) ∪⟂ (T, ⪯T) Define the following relation ⪯ on U: ∀ x, y ∈ U: x…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…nts. Let U = (S, ⪯S) ∪ (T, ⪯T) Define the following relation⟂ ⪯ on U: ∀ x, y ∈ U: x⪯y if and only if: x, y ∈ S: x ⪯ S y o…` — Gold 不切，p = 0.089，左侧 no punct.，wtp 切
- `…Let U = (S, ⪯S) ∪ (T, ⪯T) Define the following relation ⪯ on⟂ U: ∀ x, y ∈ U: x⪯y if and only if: x, y ∈ S: x ⪯ S y or: x,…` — Gold 不切，p = 0.006，左侧 no punct.，wtp 切
- `… U = (S, ⪯S) ∪ (T, ⪯T) Define the following relation ⪯ on U:⟂ ∀ x, y ∈ U: x⪯y if and only if: x, y ∈ S: x ⪯ S y or: x, y …` — Gold 不切，p = 0.077，左侧 comma/colon/dash，wtp 切
- `… = (S, ⪯S) ∪ (T, ⪯T) Define the following relation ⪯ on U: ∀⟂ x, y ∈ U: x⪯y if and only if: x, y ∈ S: x ⪯ S y or: x, y ∈ …` — Gold 不切，p = 0.061，左侧 no punct.，wtp 切
- `…⪯S) ∪ (T, ⪯T) Define the following relation ⪯ on U: ∀ x, y ∈⟂ U: x⪯y if and only if: x, y ∈ S: x ⪯ S y or: x, y ∈ T: x⪯T …` — Gold 不切，p = 0.01，左侧 no punct.，wtp 切

**14. GENTLE_dictionary_school**（dictionary）：Gold 更好 wtp F1 0.71 vs punkt 0.693；分数 0.7565 vs 0.8274；不同空隙 37 个
- `…school English Pronunciation enPR:⟂ sko͞ol, IPA(key): / skuːl / Audio (Received Pronunciation) …` — Gold 不切，p = 0.01，左侧 comma/colon/dash，wtp 切
- `…school English Pronunciation enPR: sko͞ol,⟂ IPA(key): / skuːl / Audio (Received Pronunciation) Rhymes: …` — Gold 不切，p = 0.007，左侧 comma/colon/dash，wtp 切
- `…school English Pronunciation enPR: sko͞ol, IPA(key):⟂ / skuːl / Audio (Received Pronunciation) Rhymes: -uːl Etymo…` — Gold 不切，p = 0.022，左侧 comma/colon/dash，wtp 切
- `…hool English Pronunciation enPR: sko͞ol, IPA(key): / skuːl /⟂ Audio (Received Pronunciation) Rhymes: -uːl Etymology 1 Fro…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…(key): / skuːl / Audio (Received Pronunciation) Rhymes: -uːl⟂ Etymology 1 From Middle English scole, schole (“group of pe…` — Gold 切，p = 0.541，左侧 no punct.，wtp 切
- `…ːl / Audio (Received Pronunciation) Rhymes: -uːl Etymology 1⟂ From Middle English scole, schole (“group of persons, multi…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…itude, school of fish”). Doublet of shoal. Alternative forms⟂ skull (obsolete) Noun school (1) of fish. school (plural sc…` — Gold 切，p = 0.997，左侧 no punct.，段落边界，wtp 切
- `…). Doublet of shoal. Alternative forms skull (obsolete) Noun⟂ school (1) of fish. school (plural schools) (collective) A …` — Gold 切，p = 0.994，左侧 no punct.，段落边界，wtp 切

**15. GENTLE_dictionary_trust**（dictionary）：Gold 更好 wtp F1 0.599 vs punkt 0.595；分数 0.717 vs 0.7835；不同空隙 36 个
- `…othic trausti (trausti, “alliance, pact”). Doublet of tryst.⟂ More at true, tree. Pronunciation enPR: trŭst, IPA(key): /t…` — Gold 切，p = 0.953，左侧 terminal，punkt 切
- `…ti, “alliance, pact”). Doublet of tryst. More at true, tree.⟂ Pronunciation enPR: trŭst, IPA(key): /trʌst/, [tɹʌst], [tɹɐ…` — Gold 切，p = 0.981，左侧 terminal，punkt 切
- `… pact”). Doublet of tryst. More at true, tree. Pronunciation⟂ enPR: trŭst, IPA(key): /trʌst/, [tɹʌst], [tɹɐst], [t͡ʃɹ-] (…` — Gold 切，p = 0.996，左侧 no punct.，段落边界，wtp 切
- `…on enPR: trŭst, IPA(key): /trʌst/, [tɹʌst], [tɹɐst], [t͡ʃɹ-]⟂ (Northern England) IPA(key): /trʊst/ Audio (GA) Rhymes: -ʌs…` — Gold 切，p = 0.931，左侧 no punct.，段落边界，wtp 切
- `…ɹʌst], [tɹɐst], [t͡ʃɹ-] (Northern England) IPA(key): /trʊst/⟂ Audio (GA) Rhymes: -ʌst Noun trust (countable and uncountab…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…st], [t͡ʃɹ-] (Northern England) IPA(key): /trʊst/ Audio (GA)⟂ Rhymes: -ʌst Noun trust (countable and uncountable, plural …` — Gold 切，p = 0.999，左侧 no punct.，段落边界，wtp 切
- `…: -ʌst Noun trust (countable and uncountable, plural trusts)⟂ Confidence in or reliance on some person or quality. He nee…` — Gold 切，p = 0.995，左侧 no punct.，段落边界，wtp 切
- `…trusts) Confidence in or reliance on some person or quality.⟂ He needs to regain her trust if he is ever going to win her…` — Gold 切，p = 0.995，左侧 terminal，punkt 切

**16. GENTLE_legal_service**（legal）：Gold 更好 stanza F1 0.489 vs punkt 0.489；分数 0.5113 vs 0.5149；不同空隙 24 个
- `…Exhibit 1.2 SERVICES AGREEMENT⟂ This Services Agreement ("Services Agreement" or "Agreement…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，stanza 切
- `…rnational, Inc a ___DE corporation ("Customer" or "you"). 1.⟂ Services Company agrees to make available to Customer the a…` — Gold 不切，p = 0.013，左侧 terminal，punkt 切
- `…that has established an account on the Company's website. 2.⟂ Fees and expenses a) Generally In exchange for the Services…` — Gold 不切，p = 0.004，左侧 terminal，punkt 切
- `… on the Company's website. 2. Fees and expenses a) Generally⟂ In exchange for the Services, you shall pay the Company the…` — Gold 不切，p = 0.009，左侧 no punct.，stanza 切
- `…it. Inquiries should be directed to contact@startengine.com.⟂ b) Monthly Fees and Billing The Company will bill you month…` — Gold 切，p = 0.999，左侧 terminal，段落边界，punkt 切
- `…any's transaction fees depend on the method of payment (e.g.⟂ ACH - US or WIRE - US). ACH transaction: $ 1 Wire: $ 15 d) …` — Gold 不切，p = 0.006，左侧 terminal，punkt 切
- `…l transaction. AML: $ 2 AML UK: $ 5 e) Reimbursable expenses⟂ You shall reimburse the Company for the following expenses:…` — Gold 切，p = 0.999，左侧 no punct.，段落边界，stanza 切
- `…o the Company by its third-party credit card processor. (ii)⟂ All transaction fees charged to the Company or its affiliat…` — Gold 不切，p = 0.0，左侧 no punct.，stanza 切

**17. GENTLE_esports_fifa**（esports）：Gold 更好 stanza F1 0.986 vs punkt 0.986；分数 0.974 vs 1.0；不同空隙 2 个
- `…bsolutely nothing all match. Van de Beek, Paquetá, now Theo,⟂ Theo can finally cross to the back post deep, but uh how ma…` — Gold 不切，p = 0.002，左侧 comma/colon/dash，stanza 切
- `… was just some BS. We did it we made it to the semi - finals⟂ And I just want to check out what the club objective was fo…` — Gold 切，p = 0.017，左侧 no punct.，stanza 切

**18. GENTLE_esports_fifa**（esports）：Gold 更好 sat F1 0.986 vs punkt 0.986；分数 0.9743 vs 1.0；不同空隙 2 个
- `…ing because Napoli has given me our game and Manchester City⟂ I know the way they play, they're exactly like Napoli. So w…` — Gold 不切，p = 0.011，左侧 no punct.，sat 切
- `… was just some BS. We did it we made it to the semi - finals⟂ And I just want to check out what the club objective was fo…` — Gold 切，p = 0.017，左侧 no punct.，sat 切

## lines（场景：flat）

**1. 6**（lines）：Gold 更好 sat F1 0.944 vs wtp 0.916；分数 0.9663 vs 0.973；不同空隙 24 个
- `…Silver Rhino, of course, you remember the place. Any time...⟂ we'd be very pleased... He thanked him, listening to the tw…` — Gold 不切，p = 0.651，左侧 terminal，wtp 切
- `… said uncertainly, Just a minute. I don't know about this...⟂ but the pale Cockney who was teaching him to take over his …` — Gold 不切，p = 0.791，左侧 terminal，wtp 切
- `…otograph of the Gala village football team. Mr. Kabata said,⟂ What's the matter with these people. Excuse me, I'll get a …` — Gold 不切，p = 0.157，左侧 comma/colon/dash，wtp 切
- `…a's face printed in yellow and puce on their backs. He said,⟂ All very festive, but it was distraction; he had the feelin…` — Gold 不切，p = 0.054，左侧 comma/colon/dash，wtp 切
- `…g inwardly, watching for something else. The young man said,⟂ You are from Gala district. I was. Why, are you from there?…` — Gold 不切，p = 0.063，左侧 comma/colon/dash，wtp 切
- `…the African cook's special lunch that he remembered so well:⟂ slightly burned meat soup with lots of barley, overdone ste…` — Gold 不切，p = 0.024，左侧 comma/colon/dash，wtp 切
- `… been, in the capital. There were other faces from the past;⟂ William Clough, the Governor, lifting his bristly eyebrows …` — Gold 切，p = 0.016，左侧 comma/colon/dash，wtp 切
- `…ne – we're homeless, you know... Uncle Willie's Independence⟂ Joke, Vivien said. Produces a hearty, man-of-the-world laug…` — Gold 不切，p = 0.073，左侧 no punct.，sat 切

**2. 6**（lines）：Gold 更好 punkt F1 0.938 vs wtp 0.916；分数 0.9559 vs 0.973；不同空隙 33 个
- `…He made a resigned grimace assuming understanding...⟂ My wife and I decided we couldn't stick it any longer. So w…` — Gold 不切，p = 0.999，左侧 terminal，wtp 切
- `…Silver Rhino, of course, you remember the place. Any time...⟂ we'd be very pleased... He thanked him, listening to the tw…` — Gold 不切，p = 0.651，左侧 terminal，wtp 切
- `…outhful black official at passport control said uncertainly,⟂ Just a minute. I don't know about this... but the pale Cock…` — Gold 不切，p = 0.169，左侧 comma/colon/dash，wtp 切
- `… said uncertainly, Just a minute. I don't know about this...⟂ but the pale Cockney who was teaching him to take over his …` — Gold 不切，p = 0.791，左侧 terminal，wtp 切
- `…pale Cockney who was teaching him to take over his job said,⟂ That's okay, chum, it's our ole friend Mr. Kabata. The lugg…` — Gold 不切，p = 0.134，左侧 comma/colon/dash，wtp 切
- `…otograph of the Gala village football team. Mr. Kabata said,⟂ What's the matter with these people. Excuse me, I'll get a …` — Gold 不切，p = 0.157，左侧 comma/colon/dash，wtp 切
- `…a's face printed in yellow and puce on their backs. He said,⟂ All very festive, but it was distraction; he had the feelin…` — Gold 不切，p = 0.054，左侧 comma/colon/dash，wtp 切
- `…g inwardly, watching for something else. The young man said,⟂ You are from Gala district. I was. Why, are you from there?…` — Gold 不切，p = 0.063，左侧 comma/colon/dash，wtp 切

**3. 7**（lines）：Gold 更好 wtp F1 0.913 vs punkt 0.895；分数 0.9284 vs 0.9354；不同空隙 29 个
- `…obby has heard of your greatness, sir, but of your goodness,⟂ Dobby never knew But he stopped quickly, because thinking a…` — Gold 不切，p = 0.012，左侧 comma/colon/dash，wtp 切
- `…kly, because thinking about Hermione was painful. Voldemort?⟂ said Harry. He stopped again. Thinking about Ron was painfu…` — Gold 不切，p = 0.12，左侧 terminal，punkt 切
- `…on't belong here. He is too great, too good, to lose. "Why?"⟂ said Harry in surprise. Harry Potter must not put himself i…` — Gold 不切，p = 0.013，左侧 terminal，punkt 切
- `… Potter must not put himself in peril. What terrible things?⟂ said Harry at once. Dobby made a funny choking noise and th…` — Gold 不切，p = 0.495，左侧 terminal，punkt 切
- `…nce fell downstairs. What – the – devil – are – you – doing?⟂ said Uncle Vernon through gritted teeth, his face horribly …` — Gold 不切，p = 0.869，左侧 terminal，punkt 切
- `… the wardrobe. Friends who don't even write to Harry Potter?⟂ said Dobby slyly. Dobby shuffled his feet. Dobby has them h…` — Gold 不切，p = 0.305，左侧 terminal，punkt 切
- `…need for the coming year. SECOND-YEAR STUDENTS WILL REQUIRE:⟂ The Standard Book of Spells, Grade 2 by Miranda Goshawk Bre…` — Gold 切，p = 0.806，左侧 comma/colon/dash，wtp 切
- `…OND-YEAR STUDENTS WILL REQUIRE: The Standard Book of Spells,⟂ Grade 2 by Miranda Goshawk Break with a Banshee by Gilderoy…` — Gold 不切，p = 0.006，左侧 comma/colon/dash，wtp 切

**4. 7**（lines）：Gold 更好 wtp F1 0.913 vs stanza 0.899；分数 0.9284 vs 0.9394；不同空隙 28 个
- `…obby has heard of your greatness, sir, but of your goodness,⟂ Dobby never knew But he stopped quickly, because thinking a…` — Gold 不切，p = 0.012，左侧 comma/colon/dash，wtp 切
- `…kly, because thinking about Hermione was painful. Voldemort?⟂ said Harry. He stopped again. Thinking about Ron was painfu…` — Gold 不切，p = 0.12，左侧 terminal，stanza 切
- `… Potter must not put himself in peril. What terrible things?⟂ said Harry at once. Dobby made a funny choking noise and th…` — Gold 不切，p = 0.495，左侧 terminal，stanza 切
- `…nce fell downstairs. What – the – devil – are – you – doing?⟂ said Uncle Vernon through gritted teeth, his face horribly …` — Gold 不切，p = 0.869，左侧 terminal，stanza 切
- `… the wardrobe. Friends who don't even write to Harry Potter?⟂ said Dobby slyly. Dobby shuffled his feet. Dobby has them h…` — Gold 不切，p = 0.305，左侧 terminal，stanza 切
- `… of the new books he'd need for the coming year. SECOND-YEAR⟂ STUDENTS WILL REQUIRE: The Standard Book of Spells, Grade 2…` — Gold 不切，p = 0.005，左侧 no punct.，stanza 切
- `…need for the coming year. SECOND-YEAR STUDENTS WILL REQUIRE:⟂ The Standard Book of Spells, Grade 2 by Miranda Goshawk Bre…` — Gold 切，p = 0.806，左侧 comma/colon/dash，wtp 切
- `…OND-YEAR STUDENTS WILL REQUIRE: The Standard Book of Spells,⟂ Grade 2 by Miranda Goshawk Break with a Banshee by Gilderoy…` — Gold 不切，p = 0.006，左侧 comma/colon/dash，wtp 切

**5. 3**（lines）：Gold 更好 sat F1 0.959 vs punkt 0.954；分数 0.9899 vs 0.9978；不同空隙 3 个
- `…or Cello Unaccompanied and the most intolerant perfectionist⟂ "Just as particular about music as other people are about s…` — Gold 切，p = 0.488，左侧 no punct.，sat 切
- `…upon careful staff work at the Hudson Institute. The Messrs.⟂ Kahn and Bruce Briggs say in a prefatory note that their bo…` — Gold 不切，p = 0.044，左侧 terminal，sat 切
- `…oshua, speaks about this in an excellent book of interviews,⟂ Unease in zion, edited by Ehud ben Ezer. "It is true," Yeho…` — Gold 不切，p = 0.002，左侧 comma/colon/dash，sat 切

**6. 8**（lines）：Gold 更好 stanza F1 0.988 vs sat 0.984；分数 0.9997 vs 1.0；不同空隙 1 个
- `…ry home. When she was eleven, Papa died. Within three months⟂ Mother in a little black suit, child in black warm coat, to…` — Gold 不切，p = 0.532，左侧 no punct.，sat 切

**7. 2**（lines）：Gold 更好 punkt F1 0.963 vs stanza 0.96；分数 0.9982 vs 1.0；不同空隙 1 个
- `…iotic voice that sang the same sentence over and over again:⟂ You can't make an omelette without breaking eggs. He opened…` — Gold 不切，p = 0.801，左侧 comma/colon/dash，stanza 切

**8. 2**（lines）：Gold 更好 punkt F1 0.963 vs sat 0.96；分数 0.9982 vs 1.0；不同空隙 1 个
- `…iotic voice that sang the same sentence over and over again:⟂ You can't make an omelette without breaking eggs. He opened…` — Gold 不切，p = 0.801，左侧 comma/colon/dash，sat 切

## ewt（场景：nl）

**1. reviews-390330**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2343；不同空隙 1 个
- `…od service and the reasonable prices and will definitely use⟂ American Pride Irrigation & Landscaping again.…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切

**2. reviews-390330**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2343；不同空隙 1 个
- `…od service and the reasonable prices and will definitely use⟂ American Pride Irrigation & Landscaping again.…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切

**3. reviews-390330**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2343；不同空隙 1 个
- `…od service and the reasonable prices and will definitely use⟂ American Pride Irrigation & Landscaping again.…` — Gold 不切，p = 0.001，左侧 no punct.，wtp 切

**4. reviews-388121**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.8951；不同空隙 2 个
- `…Too many kids,⟂ too many knifings, too many taserings.…` — Gold 不切，p = 0.012，左侧 comma/colon/dash，wtp 切
- `…Too many kids, too many knifings,⟂ too many taserings.…` — Gold 不切，p = 0.009，左侧 comma/colon/dash，wtp 切

**5. reviews-388121**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 0.8951；不同空隙 2 个
- `…Too many kids,⟂ too many knifings, too many taserings.…` — Gold 不切，p = 0.012，左侧 comma/colon/dash，wtp 切
- `…Too many kids, too many knifings,⟂ too many taserings.…` — Gold 不切，p = 0.009，左侧 comma/colon/dash，wtp 切

**6. reviews-388121**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.8951；不同空隙 2 个
- `…Too many kids,⟂ too many knifings, too many taserings.…` — Gold 不切，p = 0.012，左侧 comma/colon/dash，wtp 切
- `…Too many kids, too many knifings,⟂ too many taserings.…` — Gold 不切，p = 0.009，左侧 comma/colon/dash，wtp 切

**7. reviews-334808**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Great deals,⟂ great pizza!…` — Gold 不切，p = 0.007，左侧 comma/colon/dash，wtp 切

**8. reviews-334808**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Great deals,⟂ great pizza!…` — Gold 不切，p = 0.007，左侧 comma/colon/dash，wtp 切

**9. reviews-334808**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Great deals,⟂ great pizza!…` — Gold 不切，p = 0.007，左侧 comma/colon/dash，wtp 切

**10. reviews-333628**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2779；不同空隙 4 个
- `…WHAT⟂ A GREAT DEAL THANK YOU…` — Gold 不切，p = 0.003，左侧 no punct.，wtp 切
- `…WHAT A GREAT⟂ DEAL THANK YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL⟂ THANK YOU…` — Gold 不切，p = 0.113，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL THANK⟂ YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切

**11. reviews-333628**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2779；不同空隙 4 个
- `…WHAT⟂ A GREAT DEAL THANK YOU…` — Gold 不切，p = 0.003，左侧 no punct.，wtp 切
- `…WHAT A GREAT⟂ DEAL THANK YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL⟂ THANK YOU…` — Gold 不切，p = 0.113，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL THANK⟂ YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切

**12. reviews-333628**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2779；不同空隙 4 个
- `…WHAT⟂ A GREAT DEAL THANK YOU…` — Gold 不切，p = 0.003，左侧 no punct.，wtp 切
- `…WHAT A GREAT⟂ DEAL THANK YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL⟂ THANK YOU…` — Gold 不切，p = 0.113，左侧 no punct.，wtp 切
- `…WHAT A GREAT DEAL THANK⟂ YOU…` — Gold 不切，p = 0.0，左侧 no punct.，wtp 切

**13. reviews-327867**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2822；不同空隙 1 个
- `…Good⟂ clean store nice car wash…` — Gold 不切，p = 0.01，左侧 no punct.，wtp 切

**14. reviews-327867**（reviews）：Gold 更好 stanza F1 1.0 vs sat 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Good clean store⟂ nice car wash…` — Gold 不切，p = 0.035，左侧 no punct.，sat 切

**15. reviews-327867**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2822；不同空隙 1 个
- `…Good⟂ clean store nice car wash…` — Gold 不切，p = 0.01，左侧 no punct.，wtp 切

**16. reviews-327867**（reviews）：Gold 更好 punkt F1 1.0 vs sat 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Good clean store⟂ nice car wash…` — Gold 不切，p = 0.035，左侧 no punct.，sat 切

**17. reviews-314711**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Dr.⟂ White is the best!…` — Gold 不切，p = 0.256，左侧 terminal，wtp 切

**18. reviews-314711**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Dr.⟂ White is the best!…` — Gold 不切，p = 0.256，左侧 terminal，wtp 切

**19. reviews-314711**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 1.0；不同空隙 1 个
- `…Dr.⟂ White is the best!…` — Gold 不切，p = 0.256，左侧 terminal，wtp 切

**20. reviews-296357**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.0648；不同空隙 1 个
- `…Close to my house, this is the only reason⟂ I would go to this particular QT.…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

**21. reviews-296357**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 0.0648；不同空隙 1 个
- `…Close to my house, this is the only reason⟂ I would go to this particular QT.…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

**22. reviews-296357**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.0648；不同空隙 1 个
- `…Close to my house, this is the only reason⟂ I would go to this particular QT.…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

**23. reviews-295935**（reviews）：Gold 更好 stanza F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2141；不同空隙 1 个
- `…Very⟂ Informative website with alot of good work…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

**24. reviews-295935**（reviews）：Gold 更好 sat F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2141；不同空隙 1 个
- `…Very⟂ Informative website with alot of good work…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

**25. reviews-295935**（reviews）：Gold 更好 punkt F1 1.0 vs wtp 0.0；分数 0.0 vs 0.2141；不同空隙 1 个
- `…Very⟂ Informative website with alot of good work…` — Gold 不切，p = 0.002，左侧 no punct.，wtp 切

