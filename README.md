# Vis-à-vis

Conversation vidéo face à face avec un avatar IA photoréaliste, en temps réel, **entièrement sur votre machine** : aucun service cloud, aucune clé API.

Projet indépendant, inspiré du principe des applications d'avatars conversationnels ; il n'est affilié à aucun produit existant.

## Comment ça marche

```
Micro ─► détection de parole ─► Whisper ─► Ollama ─► Piper ─┐
         (Silero VAD)          (écoute)   (réponse)  (voix)  ├─► Avatar Forcing ─► navigateur
Micro et webcam (ce que l'avatar entend et voit de vous) ────┘    (visage)          (image + son)
```

| Rôle | Modèle local | Où ça tourne |
| --- | --- | --- |
| Détecter quand vous parlez | Silero VAD | processeur |
| Comprendre ce que vous dites | Whisper large-v3-turbo | carte graphique |
| Répondre | Ollama (`gemma4` par défaut) | carte graphique |
| Voix de l'avatar | Piper (2 voix FR, 2 voix EN) | processeur |
| Visage de l'avatar | Avatar Forcing | carte graphique |

Un avatar est un simple portrait. Pendant tout l'appel, le serveur génère sa vidéo sans interruption, par blocs de 10 images (0,4 s) en 512×512. Chaque bloc reçoit la suite de la voix de l'avatar (du silence s'il n'a rien à dire), ce que le micro vient d'entendre et, si la caméra est allumée, ce qu'elle vient de voir de votre visage. Le même mécanisme fait donc parler l'avatar, l'anime quand il vous écoute et quand personne ne parle.

Le navigateur affiche les images et joue la voix en se calant sur l'horloge audio. La caméra est facultative : sans elle, l'avatar ne réagit qu'à votre voix.

Vous pouvez couper la parole à l'avatar simplement en parlant (option à l'accueil), ou avec le bouton « Interrompre ».

## Lancer

Double-cliquez sur `start.bat` : il ouvre le serveur local dans une fenêtre, puis l'interface sur http://localhost:3000. Attendez « Serveur prêt » dans la fenêtre du serveur avant de démarrer une conversation.

Ollama doit être lancé. L'ensemble occupe environ 14 Go de mémoire graphique, et la génération vidéo utilise la carte en continu : fermez les applications qui s'en servent, par exemple Unreal Editor, sinon la vidéo saccade.

## Installer sur une autre machine

Prérequis : Windows, carte NVIDIA, conda, git, Node.js, Ollama avec un modèle.

```bat
server\install.bat
npm install
```

Tout le lourd (environnement Python, Avatar Forcing, modèles : environ 10 Go) va dans `D:\visavis`. Pour un autre emplacement, définissez la variable `VISAVIS_ROOT` avant `install.bat` et `run.bat`.

## Vos propres avatars

Déposez un portrait (`.png`, `.jpg`…) dans `D:\visavis\data\avatars`, puis relancez le serveur : il apparaît à l'accueil sous le nom du fichier. Une vidéo est acceptée aussi : seule sa première image sert.

Pour un bon résultat : une personne de face, bouche fermée, bien éclairée. L'image est recadrée automatiquement en carré autour du visage. Utilisez uniquement des visages que vous avez le droit d'utiliser.

La voix de chaque avatar se règle dans `D:\visavis\data\avatars\voices.json` :

```json
{ "douanier": { "fr": "fr_FR-tom-medium", "en": "en_US-ryan-medium" } }
```

Sans entrée pour un avatar, la première voix installée pour la langue est utilisée. Les voix disponibles sont listées dans `server/speech.py`.

## Réglages

Variables d'environnement lues par `server\run.bat` :

| Variable | Rôle | Défaut |
| --- | --- | --- |
| `OLLAMA_MODEL` | modèle de langage | `gemma4:latest` |
| `OLLAMA_CONTEXT` | taille du contexte du modèle (plus grand = plus de mémoire) | `8192` |
| `OLLAMA_URL` | adresse d'Ollama | `http://127.0.0.1:11434` |
| `FORCING_NFE` | étapes de calcul par bloc vidéo : plus = mouvement plus fin, mais plus lent | `7` |
| `FORCING_DRIFT_PULL` | force du rappel de la tête vers sa pose de repos, à chaque bloc | `0.08` |
| `VISAVIS_TIMINGS` | `1` pour afficher les délais de chaque étape dans la fenêtre du serveur | désactivé |
| `VISAVIS_ROOT` | dossier des modèles et de l'environnement | `D:\visavis` |
| `VISAVIS_PORT` | port du serveur local | `8765` |

La personnalité de l'avatar se règle dans `system_prompt()` et la phrase d'accueil dans `GREETINGS` (`server/main.py`).

## Deux corrections apportées au modèle

Avatar Forcing est publié comme code de recherche, pour des fichiers et non pour un flux. `server/forcing.py` le pilote bloc par bloc et corrige deux défauts gênants en conversation :

- **Mouvements de caméra.** Le modèle traduit certains mouvements par un zoom ou un glissement de toute l'image. Son mouvement se décompose en 20 composantes ; les combinaisons responsables du zoom et des glissements vertical et latéral ont été mesurées sur une vidéo générée (`server/forcing_camera.npy`) et sont maintenues à leur valeur de repos.
- **Dérive.** Laissée à elle-même, la tête finit par pencher et grossir dans l'image au bout d'une trentaine de secondes. À chaque bloc, la pose est ramenée d'une fraction vers sa position de repos.

Le modèle d'origine s'arrêtait à 41 secondes ; la limite est portée à environ 43 minutes par appel.

## Licences

Avatar Forcing est distribué sous licence CC BY-NC 4.0 : **usage non commercial uniquement**. Le dossier `server-flashhead/` contient la version précédente du serveur, fondée sur SoulX-FlashHead (sans cette restriction, image plus nette en parole, mais avatar immobile quand il écoute).

## Structure

```
server/main.py            serveur : tours de parole, Ollama, boucle vidéo, envoi image + son
server/engine.py          avatars sur disque et accès au modèle
server/forcing.py         Avatar Forcing en flux continu, corrections de pose
server/speech.py          détection de parole, Whisper, Piper
server/download_models.py téléchargement des modèles
server/test_client.py     appel de test sans navigateur
components/Conversation.tsx interface
lib/player.ts             lecteur : affichage de l'avatar calé sur l'horloge audio
public/mic-worklet.js     capture du micro en 16 kHz
server-flashhead/         version précédente du serveur (SoulX-FlashHead)
```
