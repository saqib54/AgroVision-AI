import sys
import os

# Use legacy Keras (tf_keras) for TF 2.16+ compatibility with .h5 models
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import streamlit as st
from PIL import Image

try:
    import tf_keras as keras
except ImportError:
    import tensorflow as tf
    keras = tf.keras

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Model path ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(_BASE_DIR, 'plant_disease_model.h5'))

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# ── Disease knowledge base ────────────────────────────────────────────────────
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍎',
        'disease_name': 'Apple Scab',
        'symptoms': 'Olive-green to dark-brown velvety lesions on leaves and fruit. Leaves may curl and drop early.',
        'treatment': [
            'Apply fungicides: captan, myclobutanil, or mancozeb at 7–10 day intervals.',
            'Remove and destroy all infected leaves and fallen debris immediately.',
            'Prune trees to improve air circulation around canopy.',
            'Apply lime-sulfur sprays during dormancy.',
        ],
        'prevention': [
            'Plant scab-resistant apple varieties (e.g., Liberty, Redfree).',
            'Rake and compost fallen leaves every autumn.',
            'Apply preventive fungicide spray from green-tip stage in early spring.',
            'Avoid overhead irrigation — water at soil level.',
        ],
    },
    'Apple___Black_rot': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🍎',
        'disease_name': 'Black Rot',
        'symptoms': 'Brown circular leaf spots with purple borders. Fruit shows black, sunken, mummified rot.',
        'treatment': [
            'Prune out all dead or cankered wood and burn it — do not compost.',
            'Apply captan or thiophanate-methyl fungicide every 10 days.',
            'Remove all mummified fruit from tree and ground.',
            'Disinfect pruning tools with 70% isopropyl alcohol between cuts.',
        ],
        'prevention': [
            'Maintain tree vigor with proper fertilization and irrigation.',
            'Thin fruit to reduce crowding and improve air flow.',
            'Apply protective fungicide sprays from petal fall onward.',
            'Remove fire-blight cankers — they host the black rot fungus.',
        ],
    },
    'Apple___Cedar_apple_rust': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍎',
        'disease_name': 'Cedar Apple Rust',
        'symptoms': 'Bright orange-yellow spots on upper leaf surface. Tube-like structures on leaf underside.',
        'treatment': [
            'Apply myclobutanil or propiconazole fungicide starting at bud-break.',
            'Repeat applications every 7–10 days during wet spring weather.',
            'Remove heavily infected leaves to reduce spread.',
        ],
        'prevention': [
            'Remove nearby Eastern red cedar or juniper trees (alternate host).',
            'Plant rust-resistant apple varieties.',
            'Apply protective fungicide spray from pink bud stage through petal fall.',
        ],
    },
    'Apple___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍎',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected. Leaves appear green, firm, and intact.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Continue regular monitoring every 1–2 weeks.',
            'Maintain proper fertilization and irrigation schedules.',
            'Apply preventive fungicide sprays before wet seasons.',
            'Prune annually for good air circulation.',
        ],
    },
    'Blueberry___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🫐',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms. Plant looks healthy.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Maintain soil pH between 4.5–5.5 for optimal growth.',
            'Mulch with pine bark or wood chips to retain moisture.',
            'Prune dead canes annually to encourage new growth.',
        ],
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍒',
        'disease_name': 'Powdery Mildew',
        'symptoms': 'White powdery coating on young leaves, shoots, and fruit. Leaves may curl or distort.',
        'treatment': [
            'Apply sulfur-based fungicide or potassium bicarbonate spray.',
            'Use neem oil or horticultural oil every 7–14 days.',
            'Remove and destroy heavily infected shoot tips.',
            'Avoid excess nitrogen fertilization which promotes soft growth.',
        ],
        'prevention': [
            'Plant in areas with good air circulation and full sunlight.',
            'Avoid overhead irrigation — drip irrigation preferred.',
            'Apply preventive sulfur spray before symptoms appear in humid weather.',
            'Choose powdery mildew-resistant cherry varieties.',
        ],
    },
    'Cherry_(including_sour)___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍒',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Monitor weekly during warm humid weather.',
            'Thin canopy by pruning to improve airflow.',
        ],
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🌽',
        'disease_name': 'Gray Leaf Spot',
        'symptoms': 'Rectangular gray-tan lesions with yellow halos running parallel to leaf veins. Lesions may merge causing large dead areas.',
        'treatment': [
            'Apply strobilurin or triazole fungicides (azoxystrobin, propiconazole).',
            'Start treatment at early disease appearance or at tasseling stage.',
            'Remove and destroy heavily infected plant debris after harvest.',
        ],
        'prevention': [
            'Plant resistant hybrid varieties — consult local extension service.',
            'Practice crop rotation — avoid consecutive corn planting.',
            'Till fields to bury infected residue (reduces inoculum).',
            'Ensure adequate plant spacing for good air circulation.',
        ],
    },
    'Corn_(maize)___Common_rust_': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🌽',
        'disease_name': 'Common Rust',
        'symptoms': 'Small, oval to elongated, brick-red pustules scattered across both leaf surfaces. Pustules may turn dark brown late in season.',
        'treatment': [
            'Apply fungicide (mancozeb, chlorothalonil, or strobilurins) at first sign of pustules.',
            'Treatment is most effective before tasseling.',
            'Severe infections may require 2–3 spray applications.',
        ],
        'prevention': [
            'Plant rust-resistant hybrid varieties.',
            'Plant early to avoid peak spore production periods.',
            'Monitor weather — rust spreads rapidly during cool, humid conditions.',
        ],
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🌽',
        'disease_name': 'Northern Leaf Blight',
        'symptoms': 'Long (5–15 cm), cigar-shaped, grayish-green to tan lesions running length of leaf. Can cause significant yield loss.',
        'treatment': [
            'Apply fungicides: propiconazole, azoxystrobin, or pyraclostrobin.',
            'Apply at first sign of disease, especially before tasseling.',
            'For severe cases, repeat application after 14 days.',
        ],
        'prevention': [
            'Plant resistant hybrids (Ht1, Ht2, Ht3 gene resistance).',
            'Rotate crops — do not plant corn after corn.',
            'Till to reduce surface infected residue.',
            'Avoid planting in low-lying areas with poor drainage.',
        ],
    },
    'Corn_(maize)___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🌽',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms. Healthy corn plant.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Maintain balanced soil nutrition (NPK).',
            'Scout fields weekly during the growing season.',
            'Implement crop rotation to break disease cycles.',
        ],
    },
    'Grape___Black_rot': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🍇',
        'disease_name': 'Black Rot',
        'symptoms': 'Circular brown leaf spots with black borders. Infected berries shrivel into hard, black mummies.',
        'treatment': [
            'Apply mancozeb, myclobutanil, or captan fungicide immediately.',
            'Remove and destroy all mummified berries — do NOT leave on vine.',
            'Remove infected leaves and canes during pruning.',
            'Apply fungicide every 10–14 days during wet periods.',
        ],
        'prevention': [
            'Plant resistant varieties where available.',
            'Start protective fungicide program at bud break.',
            'Train vines to improve air circulation through canopy.',
            'Remove all mummified fruit before winter.',
        ],
    },
    'Grape___Esca_(Black_Measles)': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🍇',
        'disease_name': 'Esca (Black Measles)',
        'symptoms': 'Interveinal leaf scorching (tiger stripe pattern). Berries develop dark spots. Internal wood shows dark brown streaking.',
        'treatment': [
            'No curative chemical treatment exists for Esca.',
            'Remove and destroy heavily infected vines to prevent spread.',
            'Apply sodium arsenite (where legally permitted) as wound protectant.',
            'Paint pruning wounds with fungicide paste (thiophanate-methyl).',
        ],
        'prevention': [
            'Protect pruning wounds immediately after cutting with paste fungicide.',
            'Prune during dry weather to minimize infection risk.',
            'Use double-pruning technique to delay major pruning wounds.',
            'Avoid water stress — maintain consistent irrigation.',
        ],
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍇',
        'disease_name': 'Leaf Blight',
        'symptoms': 'Irregular dark brown spots on older leaves. Infected leaves yellow and drop prematurely causing defoliation.',
        'treatment': [
            'Apply copper-based fungicide or mancozeb spray.',
            'Remove infected leaves to reduce inoculum load.',
            'Apply fungicide every 10–14 days during humid conditions.',
        ],
        'prevention': [
            'Maintain good canopy airflow through regular pruning.',
            'Avoid overhead irrigation.',
            'Apply preventive copper sprays from bud break onward.',
        ],
    },
    'Grape___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍇',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Vine is healthy! ✅'],
        'prevention': [
            'Scout vines weekly for early disease signs.',
            'Maintain canopy management (leaf removal, shoot positioning).',
            'Apply preventive copper sprays in early spring.',
        ],
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'severity': 'Critical',
        'severity_urdu': 'بحرانی',
        'color': '#dc2626',
        'icon': '🍊',
        'disease_name': 'Citrus Greening (HLB)',
        'symptoms': 'Asymmetric yellow mottling of leaves (blotchy mottle). Fruit remains green, bitter, and small. Twig dieback and tree decline.',
        'treatment': [
            '⚠️ No cure exists — infected trees cannot be saved.',
            'Remove and destroy infected trees immediately to prevent spread.',
            'Control Asian citrus psyllid (insect vector) with insecticides.',
            'Apply systemic insecticides (imidacloprid) to protect remaining trees.',
        ],
        'prevention': [
            'Use certified disease-free nursery stock only.',
            'Implement aggressive psyllid control program with insecticides.',
            'Use physical barriers (fine mesh netting) over young trees.',
            'Regularly inspect orchards — report suspected infections to authorities.',
            'Quarantine new plant material before introducing to orchard.',
        ],
    },
    'Peach___Bacterial_spot': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍑',
        'disease_name': 'Bacterial Spot',
        'symptoms': 'Small, water-soaked spots on leaves that turn brown with yellow halos. Fruit shows sunken, dark lesions.',
        'treatment': [
            'Apply copper-based bactericide (copper hydroxide, copper sulfate).',
            'Spray during dormancy and at bud swell with fixed copper.',
            'Apply oxytetracycline sprays during growing season.',
            'Remove severely infected twigs and branches.',
        ],
        'prevention': [
            'Plant resistant varieties (Contender, Reliance, Redhaven).',
            'Avoid planting in frost-prone areas (frost wounds allow bacterial entry).',
            'Apply copper sprays from petal fall through fruit development.',
            'Reduce tree stress through proper fertilization and irrigation.',
        ],
    },
    'Peach___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍑',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Monitor weekly for bacterial spot symptoms.',
            'Apply preventive copper sprays before rainy periods.',
            'Thin fruit to reduce disease pressure.',
        ],
    },
    'Pepper,_bell___Bacterial_spot': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🫑',
        'disease_name': 'Bacterial Spot',
        'symptoms': 'Small, water-soaked spots on leaves turning brown with yellow halos. Fruit develops raised, scab-like lesions.',
        'treatment': [
            'Apply copper bactericide sprays (copper hydroxide or copper sulfate).',
            'Spray every 5–7 days during wet weather.',
            'Remove heavily infected plant parts.',
            'Avoid working with plants when wet to prevent spread.',
        ],
        'prevention': [
            'Use certified disease-free transplants and seeds.',
            'Practice 3-year crop rotation with non-solanaceous crops.',
            'Avoid overhead irrigation — use drip irrigation.',
            'Disinfect tools with 10% bleach solution between uses.',
        ],
    },
    'Pepper,_bell___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🫑',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Keep plants well-watered and fertilized.',
            'Monitor weekly for bacterial spot signs.',
        ],
    },
    'Potato___Early_blight': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🥔',
        'disease_name': 'Early Blight',
        'symptoms': "Dark brown, circular spots with concentric rings (bull's-eye pattern) on older leaves. Yellowing around spots.",
        'treatment': [
            'Apply chlorothalonil, mancozeb, or azoxystrobin fungicide.',
            'Start spraying at first symptom appearance.',
            'Spray every 7–10 days, especially during warm humid weather.',
            'Remove and destroy infected leaves.',
        ],
        'prevention': [
            'Use certified disease-free seed potatoes.',
            'Practice 3–4 year crop rotation.',
            'Ensure adequate plant nutrition — stressed plants are more susceptible.',
            'Hill plants to prevent soil splash spreading spores.',
        ],
    },
    'Potato___Late_blight': {
        'severity': 'Critical',
        'severity_urdu': 'بحرانی',
        'color': '#dc2626',
        'icon': '🥔',
        'disease_name': 'Late Blight',
        'symptoms': 'Water-soaked, dark green to brown lesions on leaves. White fuzzy mold on leaf undersides in humid conditions. Tubers show dark rotten areas.',
        'treatment': [
            '⚠️ Act immediately — Late blight spreads extremely fast.',
            'Apply mancozeb, chlorothalonil, or cymoxanil-based fungicide.',
            'Spray every 5–7 days during cool wet weather.',
            'Remove and destroy ALL infected plant material — bag and discard.',
            'Do not compost infected material.',
        ],
        'prevention': [
            'Plant resistant varieties (Sarpo Mira, Defender, Elba).',
            'Use certified disease-free seed potatoes.',
            'Apply preventive fungicide before first symptoms appear.',
            'Ensure good field drainage — avoid waterlogging.',
            'Monitor blight forecast services to time spray applications.',
        ],
    },
    'Potato___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🥔',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Scout fields weekly, especially after rainy periods.',
            'Maintain soil fertility and balanced nutrition.',
            'Hill soil around plants to protect tubers.',
        ],
    },
    'Raspberry___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🫐',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Prune out old canes after fruiting.',
            'Maintain good air circulation.',
            'Apply mulch to prevent soil splash.',
        ],
    },
    'Soybean___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🌿',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Rotate with non-legume crops every 2–3 years.',
            'Monitor fields weekly during vegetative and reproductive stages.',
        ],
    },
    'Squash___Powdery_mildew': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🥦',
        'disease_name': 'Powdery Mildew',
        'symptoms': 'White powdery coating on upper and lower leaf surfaces. Leaves yellow and die as disease progresses.',
        'treatment': [
            'Apply potassium bicarbonate, neem oil, or sulfur fungicide spray.',
            'Spray every 7–10 days until symptoms subside.',
            'Remove and dispose of severely infected leaves.',
            'Avoid excess nitrogen fertilizer which promotes soft susceptible growth.',
        ],
        'prevention': [
            'Plant resistant squash varieties.',
            'Ensure wide plant spacing for good airflow.',
            'Water at soil level — avoid wetting foliage.',
            'Apply neem oil spray preventively during warm dry weather.',
        ],
    },
    'Strawberry___Leaf_scorch': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍓',
        'disease_name': 'Leaf Scorch',
        'symptoms': 'Irregular purple to dark brown spots on upper leaf surface. Centers turn gray-tan. Leaf edges appear scorched.',
        'treatment': [
            'Apply captan or myclobutanil fungicide.',
            'Spray every 7–14 days during wet weather.',
            'Remove and destroy old infected leaves.',
            'Renovate beds after harvest — mow, thin, and fertilize.',
        ],
        'prevention': [
            'Plant certified disease-free transplants.',
            'Avoid overcrowding — maintain proper plant spacing.',
            'Use drip irrigation to keep foliage dry.',
            'Apply straw mulch to prevent soil splash.',
        ],
    },
    'Strawberry___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍓',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Renovate beds after fruiting for next season.',
            'Monitor for early disease signs weekly.',
        ],
    },
    'Tomato___Bacterial_spot': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Bacterial Spot',
        'symptoms': 'Small, water-soaked spots on leaves turning brown with yellow halos. Fruit shows raised, scabby lesions.',
        'treatment': [
            'Apply copper bactericide spray (copper hydroxide).',
            'Spray every 5–7 days during wet conditions.',
            'Remove infected leaves and stems.',
            'Avoid overhead irrigation.',
        ],
        'prevention': [
            'Use resistant varieties and certified disease-free seed.',
            'Practice 3-year rotation with non-solanaceous crops.',
            'Sanitize tools with 10% bleach solution.',
            'Stake and prune plants for good air circulation.',
        ],
    },
    'Tomato___Early_blight': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Early Blight',
        'symptoms': "Dark brown spots with concentric rings (target/bull's-eye pattern) on older leaves. Yellow halo surrounds lesions.",
        'treatment': [
            'Apply chlorothalonil, mancozeb, or azoxystrobin fungicide.',
            'Spray every 7–10 days during warm humid periods.',
            'Remove infected lower leaves and destroy them.',
            'Mulch heavily to prevent soil splash inoculation.',
        ],
        'prevention': [
            'Rotate crops — avoid tomatoes in same spot for 3 years.',
            'Use disease-resistant varieties.',
            'Water at base of plant with drip irrigation.',
            'Maintain healthy plants — stressed plants are more susceptible.',
        ],
    },
    'Tomato___Late_blight': {
        'severity': 'Critical',
        'severity_urdu': 'بحرانی',
        'color': '#dc2626',
        'icon': '🍅',
        'disease_name': 'Late Blight',
        'symptoms': 'Large, greasy, water-soaked dark lesions on leaves and stems. White fuzzy mold on undersides in high humidity. Fruit turns brown and rots.',
        'treatment': [
            '⚠️ Treat immediately — spreads in 24–48 hours under wet conditions.',
            'Apply mancozeb, chlorothalonil, or cymoxanil fungicide immediately.',
            'Remove and bag all infected material — do NOT compost.',
            'Spray every 5–7 days continuing until dry weather.',
        ],
        'prevention': [
            'Plant resistant varieties (Defiant, Mountain Magic, Juliet).',
            'Avoid overhead irrigation.',
            'Monitor blight alerts for your region.',
            'Apply preventive copper or mancozeb spray before wet weather.',
            'Improve field drainage.',
        ],
    },
    'Tomato___Leaf_Mold': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Leaf Mold',
        'symptoms': 'Yellow spots on upper leaf surface. Olive-green to grayish-brown velvety mold on leaf undersides.',
        'treatment': [
            'Apply chlorothalonil, mancozeb, or copper fungicide.',
            'Improve greenhouse ventilation immediately to reduce humidity.',
            'Remove and destroy infected leaves.',
            'Spray every 7–14 days until controlled.',
        ],
        'prevention': [
            'Maintain relative humidity below 85% in greenhouses.',
            'Use resistant varieties with Cf resistance genes.',
            'Ensure wide plant spacing and pruning for air circulation.',
            'Avoid overhead misting systems.',
        ],
    },
    'Tomato___Septoria_leaf_spot': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Septoria Leaf Spot',
        'symptoms': 'Numerous small, circular spots with dark borders and light gray centers on lower leaves. Tiny dark specks (pycnidia) visible in lesion centers.',
        'treatment': [
            'Apply chlorothalonil, mancozeb, or copper-based fungicide.',
            'Spray every 7–10 days during wet weather.',
            'Remove infected lower leaves immediately.',
            'Avoid wetting foliage when watering.',
        ],
        'prevention': [
            'Practice 3-year crop rotation.',
            'Use mulch to prevent soil splash.',
            'Use drip irrigation.',
            'Stake plants for better air flow.',
        ],
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Spider Mites',
        'symptoms': 'Fine stippling or bronzing on leaf surface. Webbing on leaf undersides. Leaves turn yellow then brown and drop.',
        'treatment': [
            'Apply miticide: abamectin, bifenazate, or spiromesifen.',
            'Use neem oil or insecticidal soap spray for mild infestations.',
            'Spray leaf undersides thoroughly — mites live there.',
            'Repeat treatment every 5–7 days for 2–3 applications.',
        ],
        'prevention': [
            'Avoid plant water stress — drought favors mite outbreaks.',
            'Introduce predatory mites (Phytoseiidae) as biological control.',
            'Avoid excessive nitrogen fertilization.',
            'Regularly mist plants to increase humidity (mites prefer hot dry conditions).',
        ],
    },
    'Tomato___Target_Spot': {
        'severity': 'Moderate',
        'severity_urdu': 'درمیانہ',
        'color': '#f59e0b',
        'icon': '🍅',
        'disease_name': 'Target Spot',
        'symptoms': 'Brown spots with concentric rings and yellow margins on leaves. Fruit develops sunken, brown, circular lesions.',
        'treatment': [
            'Apply azoxystrobin, chlorothalonil, or mancozeb fungicide.',
            'Start treatment at first sign of disease.',
            'Spray every 7–14 days during warm humid conditions.',
            'Remove and destroy heavily infected plant material.',
        ],
        'prevention': [
            'Use drip irrigation — keep foliage dry.',
            'Prune lower leaves to improve airflow at canopy base.',
            'Practice crop rotation with non-solanaceous crops.',
            'Apply preventive fungicide sprays at canopy closure.',
        ],
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'severity': 'Critical',
        'severity_urdu': 'بحرانی',
        'color': '#dc2626',
        'icon': '🍅',
        'disease_name': 'Yellow Leaf Curl Virus',
        'symptoms': 'Leaves curl upward and inward with yellow margins. Severe stunting and bushy appearance. Little to no fruit production.',
        'treatment': [
            '⚠️ No cure — virus cannot be eliminated from infected plants.',
            'Remove and destroy infected plants immediately to prevent spread.',
            'Control whitefly vector with insecticides (imidacloprid, thiamethoxam).',
            'Use yellow sticky traps to monitor whitefly populations.',
        ],
        'prevention': [
            'Plant resistant/tolerant varieties (TY varieties).',
            'Use insect-proof nets (50-mesh) over seedling beds.',
            'Apply systemic insecticide at planting to control whiteflies.',
            'Eliminate weed hosts of whitefly near the field.',
            'Remove volunteer tomato plants from the area.',
        ],
    },
    'Tomato___Tomato_mosaic_virus': {
        'severity': 'High',
        'severity_urdu': 'زیادہ',
        'color': '#ef4444',
        'icon': '🍅',
        'disease_name': 'Mosaic Virus',
        'symptoms': 'Mosaic pattern of light and dark green on leaves. Leaves may be narrow and crinkled. Fruit shows yellow blotching.',
        'treatment': [
            '⚠️ No chemical cure for viral infection.',
            'Remove and destroy infected plants.',
            'Wash hands thoroughly after handling infected plants.',
            'Control aphid vectors with insecticidal soap or neem oil.',
        ],
        'prevention': [
            'Use certified virus-free seed or resistant varieties.',
            'Disinfect all tools with 10% bleach before and after use.',
            'Wash hands before handling plants — virus spreads mechanically.',
            'Control aphid populations which transmit the virus.',
            'Remove and destroy infected plant material immediately.',
        ],
    },
    'Tomato___healthy': {
        'severity': 'None',
        'severity_urdu': 'صفر',
        'color': '#22c55e',
        'icon': '🍅',
        'disease_name': 'Healthy',
        'symptoms': 'No disease symptoms detected. Plant appears healthy.',
        'treatment': ['No treatment needed. Plant is healthy! ✅'],
        'prevention': [
            'Continue regular watering and fertilization schedule.',
            'Scout weekly for early disease signs.',
            'Stake and prune for good air circulation.',
            'Apply preventive copper spray before wet seasons.',
        ],
    },
}


def get_disease_info(class_name: str) -> dict:
    """Return disease info dict for a given class name."""
    return DISEASE_INFO.get(class_name, {
        'severity': 'Unknown',
        'severity_urdu': 'نامعلوم',
        'color': '#94a3b8',
        'icon': '🌿',
        'disease_name': 'Unknown',
        'symptoms': 'No additional information available.',
        'treatment': ['Consult a local agricultural extension officer.'],
        'prevention': ['Monitor plant regularly for changes.'],
    })


# ── Fix: FixedDropout is a custom layer from EfficientNet / older keras ───────
class FixedDropout(keras.layers.Dropout):
    """EfficientNet's FixedDropout – identical to Dropout but noise shape is
    resolved against the *static* input shape, not the dynamic one."""
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple(
            inputs.shape[i] if sh is None else sh
            for i, sh in enumerate(self.noise_shape)
        )


@st.cache_resource(show_spinner=False)
def load_model_and_size():
    """Load the model using tf_keras for TF 2.16+ compatibility with .h5 models."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at:\n`{MODEL_PATH}`")
        st.stop()
    with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
        model = keras.models.load_model(MODEL_PATH)
    _, h, w, _ = model.input_shape
    return model, (h, w)


def preprocess_image(image: Image.Image, size: tuple) -> np.ndarray:
    """Resize to whatever size the model actually expects, then normalise."""
    image = image.resize((size[1], size[0]))   # PIL uses (width, height)
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)


def format_label(raw: str) -> tuple:
    """Return (plant, condition) tuple from a CLASS_NAME string."""
    parts = raw.split('___', 1)
    plant = parts[0].replace('_', ' ').replace(',', ',')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    return plant, condition


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1f1a 50%, #0f1a2e 100%);
    min-height: 100vh;
}

/* ── header hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.hero p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin: 0;
}

/* ── glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
}

/* ── DIAGNOSIS RESULT BANNER ── */
.diagnosis-banner {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.diagnosis-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 20px;
    opacity: 0.08;
    z-index: 0;
}
.diagnosis-content { position: relative; z-index: 1; }

.dx-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.dx-plant {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
.dx-disease {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.dx-disease.healthy { color: #4ade80; }
.dx-disease.diseased { color: #fb923c; }

.dx-badges {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.dx-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── confidence bar ── */
.conf-label {
    color: #64748b;
    font-size: 0.78rem;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.conf-value {
    color: #e2e8f0;
    font-weight: 700;
}
.conf-bar-outer {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
}

/* ── section cards ── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── symptom text ── */
.symptom-text {
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 0.95rem;
    margin: 0;
}

/* ── treatment steps ── */
.treatment-list {
    list-style: none;
    margin: 0;
    padding: 0;
}
.treatment-item {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin-bottom: 0.8rem;
    padding: 0.9rem 1rem;
    background: rgba(251,146,60,0.06);
    border: 1px solid rgba(251,146,60,0.15);
    border-radius: 12px;
    color: #fde68a;
    font-size: 0.9rem;
    line-height: 1.5;
}
.treatment-num {
    background: rgba(251,146,60,0.2);
    color: #fb923c;
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.05rem;
}

/* ── prevention steps ── */
.prevention-list {
    list-style: none;
    margin: 0;
    padding: 0;
}
.prevention-item {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin-bottom: 0.7rem;
    color: #a7f3d0;
    font-size: 0.9rem;
    line-height: 1.5;
}
.prevention-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4ade80;
    flex-shrink: 0;
    margin-top: 0.45rem;
}

/* ── uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(74,222,128,0.3) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
}

/* ── button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4ade80, #22d3ee) !important;
    color: #0a0f1e !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── info chips ── */
.chip {
    display: inline-block;
    background: rgba(74,222,128,0.1);
    color: #4ade80;
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── model info card ── */
.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}
.model-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
}
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.8rem;
}
.model-stat {
    background: rgba(74,222,128,0.05);
    border: 1px solid rgba(74,222,128,0.12);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    text-align: center;
}
.model-stat-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #4ade80;
    display: block;
    margin-bottom: 0.2rem;
}
.model-stat-label {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 500;
}

/* ── top predictions ── */
.top-pred {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
}
.pred-label {
    color: #cbd5e1;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
}
.pred-pct { color: #4ade80; font-weight: 700; }
</style>
"""


def render_diagnosis(predicted_class, confidence, predictions):
    """Render the full diagnosis panel with disease name, symptoms, treatment, prevention."""
    plant, condition = format_label(predicted_class)
    info = get_disease_info(predicted_class)

    is_healthy = 'healthy' in condition.lower()
    disease_class = 'healthy' if is_healthy else 'diseased'
    status_icon = '✅' if is_healthy else '🔴'
    severity = info.get('severity', 'Unknown')
    sev_color = info.get('color', '#94a3b8')
    disease_name = info.get('disease_name', condition)
    plant_icon = info.get('icon', '🌿')
    conf_int = int(confidence)

    # ── MAIN DIAGNOSIS BANNER ──────────────────────────────────────────────
    sev_bg = f"{sev_color}15"
    sev_border = f"{sev_color}40"
    st.markdown(f"""
    <div class="diagnosis-banner" style="background:{sev_bg}; border:1.5px solid {sev_border};">
        <div class="diagnosis-content">
            <div class="dx-label">🔬 Diagnosis Result</div>
            <div class="dx-plant">{plant_icon} &nbsp; {plant}</div>
            <div class="dx-disease {disease_class}">{disease_name}</div>

            <div class="dx-badges">
                <span class="dx-badge" style="background:{sev_color}20; color:{sev_color}; border:1px solid {sev_color}50;">
                    {'✅ Healthy' if is_healthy else f'⚠️ {severity} Severity'}
                </span>
                <span class="dx-badge" style="background:rgba(34,211,238,0.12); color:#22d3ee; border:1px solid rgba(34,211,238,0.3);">
                    🎯 {confidence:.1f}% Confidence
                </span>
            </div>

            <div class="conf-label">AI Confidence Score &nbsp;·&nbsp; <span class="conf-value">{confidence:.1f}%</span></div>
            <div class="conf-bar-outer">
                <div class="conf-bar-inner" style="width:{conf_int}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SYMPTOMS ──────────────────────────────────────────────────────────
    symptoms_text = info.get('symptoms', 'N/A')
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background:rgba(99,102,241,0.15);">🔍</div>
            <div class="section-title" style="color:#818cf8;">What We Detected</div>
        </div>
        <p class="symptom-text">{symptoms_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TREATMENT ─────────────────────────────────────────────────────────
    treatment_list = info.get('treatment', [])
    treatment_html = ''
    for i, t in enumerate(treatment_list):
        treatment_html += f"""
        <li class="treatment-item">
            <span class="treatment-num">{i+1}</span>
            <span>{t}</span>
        </li>"""

    heading = '💊 Treatment Steps' if not is_healthy else '✅ No Treatment Needed'
    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background:rgba(251,146,60,0.15);">💊</div>
            <div class="section-title" style="color:#fb923c;">{heading}</div>
        </div>
        <ul class="treatment-list">
            {treatment_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── PREVENTION ────────────────────────────────────────────────────────
    prevention_list = info.get('prevention', [])
    prevention_html = ''
    for p in prevention_list:
        prevention_html += f"""
        <li class="prevention-item">
            <span class="prevention-dot"></span>
            <span>{p}</span>
        </li>"""

    st.markdown(f"""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background:rgba(74,222,128,0.12);">🛡️</div>
            <div class="section-title" style="color:#4ade80;">Prevention Tips</div>
        </div>
        <ul class="prevention-list">
            {prevention_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── TOP 3 PREDICTIONS ─────────────────────────────────────────────────
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    with st.expander("📊 Top 3 Predictions"):
        rank_icons = ['🥇', '🥈', '🥉']
        for rank, idx in enumerate(top3_idx):
            p, c = format_label(CLASS_NAMES[idx])
            pct = float(predictions[0][idx]) * 100
            bar = int(pct)
            inf = get_disease_info(CLASS_NAMES[idx])
            dn = inf.get('disease_name', c)
            st.markdown(f"""
            <div class="top-pred">
                <div class="pred-label">
                    {rank_icons[rank]} &nbsp; <b>{p}</b> — {dn} &nbsp;
                    <span class="pred-pct">{pct:.1f}%</span>
                </div>
                <div class="conf-bar-outer">
                    <div class="conf-bar-inner" style="width:{bar}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AgroVision AI — Plant Disease Detector",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🌿 AgroVision AI</h1>
        <p>Upload a leaf image — AI detects disease & gives complete treatment solution</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Chips ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.4rem;">
        <span class="chip">🧠 EfficientNet-B0</span>
        <span class="chip">🌱 38 Disease Classes</span>
        <span class="chip">🎯 94.99% Accuracy</span>
        <span class="chip">💊 Treatment & Prevention</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Info Card ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="model-card">
        <div class="model-card-title">🔬 Model Information</div>
        <div class="model-grid">
            <div class="model-stat">
                <span class="model-stat-value">97.8%</span>
                <span class="model-stat-label">Train Accuracy</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-value">94.99%</span>
                <span class="model-stat-label">Val Accuracy</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-value">87K</span>
                <span class="model-stat-label">Training Images</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-value">38</span>
                <span class="model-stat-label">Disease Classes</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-value">512²</span>
                <span class="model-stat-label">Input Size</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-value">50</span>
                <span class="model-stat-label">Epochs Trained</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ───────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Drop a plant leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col_img, col_btn = st.columns([2, 1], gap="large")
        with col_img:
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
        with col_btn:
            st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem; line-height:1.6;">
                📸 Image loaded successfully.<br>
                Click the button below to run AI analysis.
            </div>
            """, unsafe_allow_html=True)
            predict_clicked = st.button("🔍  Analyse Leaf", use_container_width=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Prediction ───────────────────────────────────────────────────────
        if predict_clicked:
            with st.spinner("🧠 Loading AI model & analysing leaf…"):
                model, img_size = load_model_and_size()
                img_array = preprocess_image(image, img_size)
                predictions = model.predict(img_array, verbose=0)
                result_index = int(np.argmax(predictions[0]))
                predicted_class = CLASS_NAMES[result_index]
                confidence = float(predictions[0][result_index]) * 100

            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
            render_diagnosis(predicted_class, confidence, predictions)

        else:
            st.markdown("""
            <div class="section-card" style="text-align:center; color:#475569; padding:2.5rem 1rem;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔬</div>
                <div style="font-size:1rem; color:#64748b;">Click <b style="color:#4ade80;">Analyse Leaf</b> to get AI diagnosis</div>
                <div style="font-size:0.82rem; color:#475569; margin-top:0.4rem;">
                    Get: Disease Name · Symptoms · Treatment Steps · Prevention Tips
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="section-card" style="text-align:center; color:#64748b; padding:3rem 1rem;">
            <div style="font-size:4rem; margin-bottom:0.8rem;">🍃</div>
            <div style="font-size:1.1rem; font-weight:600; color:#94a3b8; margin-bottom:0.4rem;">
                Upload a leaf image to get started
            </div>
            <div style="font-size:0.85rem; color:#475569;">
                Supports JPG, JPEG, PNG formats<br>
                AI will detect disease & provide complete treatment solution
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.78rem; line-height:1.8;">
        🌿 <b style="color:#4ade80;">AgroVision AI</b>&nbsp;·&nbsp;
        Model: <b style="color:#22d3ee;">EfficientNet-B0</b>&nbsp;·&nbsp;
        Dataset: <b style="color:#a78bfa;">PlantVillage (~87K images)</b><br>
        Powered by TensorFlow &amp; Streamlit &nbsp;·&nbsp;
        <a href="https://github.com/saqib54/AgroVision-AI" style="color:#4ade80; text-decoration:none;">
            GitHub ↗
        </a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
