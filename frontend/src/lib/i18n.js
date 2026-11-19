/** Labels for the plant result in English, Hindi and Kannada. The AI-written text comes translated from the backend. */

export const LANGS = [
  ['en', 'English'],
  ['hi', 'हिन्दी'],
  ['kn', 'ಕನ್ನಡ'],
]

const STRINGS = {
  en: {
    language: 'Language',
    fromPlantNet: 'from PlantNet',
    bestMatch: (n) => `Best match from ${n} photo${n === 1 ? '' : 's'}`,
    family: 'Family',
    confidence: 'PlantNet confidence',
    lowTitle: 'Low confidence',
    lowBody: 'Treat this as a candidate, not an identification. Try a sharper, closer photo, or add more views of the same plant.',
    others: 'Other possible plants',
    othersCaption: 'Other PlantNet candidates',
    disclaimer:
      'A photo match is not botanical confirmation, and the uses are written by AI, not a doctor. Don’t eat or use a plant based on this result. Check it with a botanist and ask a doctor before any medicinal use.',
    usesTitle: 'Medicinal uses',
    writtenByAI: 'written by AI',
    fromGuide: 'from app guide',
    localName: '',
    loadingUses: 'Loading medicinal uses',
    usesFailed: 'Medicinal uses couldn’t be loaded',
    noAI: 'No AI service is set up on this server.',
    busy: 'The AI is busy right now. Try again in a minute.',
    tryAgain: 'Try again',
    notTranslated: 'The translation isn’t available right now, so this is in English.',
    toxicTitle: 'This plant is poisonous',
    toxicBody: 'Don’t eat it or make remedies from it at home. See the safety notes below.',
    noUse: 'No known medicinal use for this plant.',
    partsUsed: 'Parts used: ',
    safety: 'Safety',
    evidence: { moderate: 'studied in people', limited: 'limited studies', traditional: 'traditional use' },
  },
  hi: {
    language: 'भाषा',
    fromPlantNet: 'PlantNet से',
    bestMatch: (n) => `${n} फ़ोटो से सबसे सही मिलान`,
    family: 'कुल',
    confidence: 'PlantNet का भरोसा',
    lowTitle: 'कम भरोसा',
    lowBody: 'इसे पक्की पहचान नहीं, सिर्फ़ एक संभावना मानें। ज़्यादा साफ़ और पास से ली गई फ़ोटो आज़माएँ, या उसी पौधे की और फ़ोटो जोड़ें।',
    others: 'अन्य संभावित पौधे',
    othersCaption: 'PlantNet के अन्य संभावित पौधे',
    disclaimer:
      'फ़ोटो से मिलान पौधे की वैज्ञानिक पुष्टि नहीं है, और ये उपयोग AI ने लिखे हैं, डॉक्टर ने नहीं। इस नतीजे के आधार पर कोई पौधा न खाएँ और न इस्तेमाल करें। किसी वनस्पति विशेषज्ञ से इसकी जाँच कराएँ और औषधीय उपयोग से पहले डॉक्टर से पूछें।',
    usesTitle: 'औषधीय उपयोग',
    writtenByAI: 'AI द्वारा लिखा गया',
    fromGuide: 'ऐप गाइड से',
    localName: 'हिंदी नाम',
    loadingUses: 'औषधीय उपयोग लोड हो रहे हैं',
    usesFailed: 'औषधीय उपयोग लोड नहीं हो सके',
    noAI: 'इस सर्वर पर कोई AI सेवा सेट नहीं है।',
    busy: 'AI अभी व्यस्त है। एक मिनट बाद फिर कोशिश करें।',
    tryAgain: 'फिर कोशिश करें',
    notTranslated: 'अभी अनुवाद उपलब्ध नहीं है, इसलिए यह अंग्रेज़ी में दिखाया गया है।',
    toxicTitle: 'यह पौधा ज़हरीला है',
    toxicBody: 'इसे न खाएँ और न ही घर पर इससे कोई नुस्खा बनाएँ। नीचे दी गई सुरक्षा जानकारी देखें।',
    noUse: 'इस पौधे का कोई ज्ञात औषधीय उपयोग नहीं है।',
    partsUsed: 'उपयोग होने वाले भाग: ',
    safety: 'सुरक्षा',
    evidence: { moderate: 'लोगों पर अध्ययन हुआ', limited: 'सीमित अध्ययन', traditional: 'पारंपरिक उपयोग' },
  },
  kn: {
    language: 'ಭಾಷೆ',
    fromPlantNet: 'PlantNet ನಿಂದ',
    bestMatch: (n) => `${n} ಫೋಟೋದಿಂದ ಅತ್ಯುತ್ತಮ ಹೊಂದಾಣಿಕೆ`,
    family: 'ಕುಟುಂಬ',
    confidence: 'PlantNet ವಿಶ್ವಾಸ',
    lowTitle: 'ಕಡಿಮೆ ವಿಶ್ವಾಸ',
    lowBody: 'ಇದನ್ನು ಖಚಿತ ಗುರುತು ಎಂದು ಅಲ್ಲ, ಕೇವಲ ಒಂದು ಸಾಧ್ಯತೆ ಎಂದು ಪರಿಗಣಿಸಿ. ಹೆಚ್ಚು ಸ್ಪಷ್ಟವಾದ, ಹತ್ತಿರದಿಂದ ತೆಗೆದ ಫೋಟೋ ಪ್ರಯತ್ನಿಸಿ, ಅಥವಾ ಅದೇ ಸಸ್ಯದ ಇನ್ನಷ್ಟು ಫೋಟೋಗಳನ್ನು ಸೇರಿಸಿ.',
    others: 'ಇತರ ಸಂಭಾವ್ಯ ಸಸ್ಯಗಳು',
    othersCaption: 'PlantNet ನ ಇತರ ಸಂಭಾವ್ಯ ಸಸ್ಯಗಳು',
    disclaimer:
      'ಫೋಟೋ ಹೊಂದಾಣಿಕೆ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ದೃಢೀಕರಣವಲ್ಲ, ಮತ್ತು ಈ ಉಪಯೋಗಗಳನ್ನು ಬರೆದಿರುವುದು AI, ವೈದ್ಯರಲ್ಲ. ಈ ಫಲಿತಾಂಶದ ಆಧಾರದ ಮೇಲೆ ಯಾವುದೇ ಸಸ್ಯವನ್ನು ತಿನ್ನಬೇಡಿ ಅಥವಾ ಬಳಸಬೇಡಿ. ಸಸ್ಯಶಾಸ್ತ್ರಜ್ಞರಿಂದ ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಯಾವುದೇ ಔಷಧೀಯ ಬಳಕೆಗೆ ಮೊದಲು ವೈದ್ಯರನ್ನು ಕೇಳಿ.',
    usesTitle: 'ಔಷಧೀಯ ಉಪಯೋಗಗಳು',
    writtenByAI: 'AI ಬರೆದಿದೆ',
    fromGuide: 'ಆ್ಯಪ್ ಮಾರ್ಗದರ್ಶಿಯಿಂದ',
    localName: 'ಕನ್ನಡ ಹೆಸರು',
    loadingUses: 'ಔಷಧೀಯ ಉಪಯೋಗಗಳನ್ನು ಲೋಡ್ ಮಾಡಲಾಗುತ್ತಿದೆ',
    usesFailed: 'ಔಷಧೀಯ ಉಪಯೋಗಗಳನ್ನು ಲೋಡ್ ಮಾಡಲು ಆಗಲಿಲ್ಲ',
    noAI: 'ಈ ಸರ್ವರ್‌ನಲ್ಲಿ ಯಾವುದೇ AI ಸೇವೆ ಹೊಂದಿಸಿಲ್ಲ.',
    busy: 'AI ಈಗ ಕಾರ್ಯನಿರತವಾಗಿದೆ. ಒಂದು ನಿಮಿಷದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
    tryAgain: 'ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ',
    notTranslated: 'ಈಗ ಅನುವಾದ ಲಭ್ಯವಿಲ್ಲ, ಆದ್ದರಿಂದ ಇದನ್ನು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ.',
    toxicTitle: 'ಈ ಸಸ್ಯ ವಿಷಕಾರಿ',
    toxicBody: 'ಇದನ್ನು ತಿನ್ನಬೇಡಿ ಅಥವಾ ಮನೆಯಲ್ಲಿ ಇದರಿಂದ ಮದ್ದು ತಯಾರಿಸಬೇಡಿ. ಕೆಳಗಿನ ಸುರಕ್ಷತಾ ಸೂಚನೆಗಳನ್ನು ನೋಡಿ.',
    noUse: 'ಈ ಸಸ್ಯಕ್ಕೆ ಯಾವುದೇ ತಿಳಿದಿರುವ ಔಷಧೀಯ ಉಪಯೋಗವಿಲ್ಲ.',
    partsUsed: 'ಬಳಸುವ ಭಾಗಗಳು: ',
    safety: 'ಸುರಕ್ಷತೆ',
    evidence: { moderate: 'ಜನರ ಮೇಲೆ ಅಧ್ಯಯನ ಮಾಡಲಾಗಿದೆ', limited: 'ಸೀಮಿತ ಅಧ್ಯಯನಗಳು', traditional: 'ಸಾಂಪ್ರದಾಯಿಕ ಬಳಕೆ' },
  },
}

/** The label set for a language (English for anything unknown). */
export const strings = (lang) => STRINGS[lang] ?? STRINGS.en

const KEY = 'plant-lang'

/** The language picked last time; storage can be blocked, so fall back to English. */
export function savedLang() {
  try {
    const v = localStorage.getItem(KEY)
    return STRINGS[v] ? v : 'en'
  } catch {
    return 'en'
  }
}

export function saveLang(lang) {
  try {
    localStorage.setItem(KEY, lang)
  } catch {
    // Not remembered; the page still works.
  }
}
