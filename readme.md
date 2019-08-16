# nlp-intro

Dette repeot inneholder en presentasjon som er ment å gi en introduksjon til naturlig
språkrpsoessering (NLP): Fokuset i presentasjonen er på:
- Teknikker for å representere tekst på et format som gjør den egent for maskinlæring
- Teknikker for å forhåndsprosssesere tekst
- Teknikker for å visualisere tekst, og resultater fra tekst-modeller
 
  
 # Oppsett
 
  Oppsettet er testet på OSX og Windows. Er du på Linux vil dette mest
  sannsynligvis funke.
  
 ## Anaconda
 
 Anaconda er en python-distribusjon som er spesielt egnet for data-arbeid.
 Anaconda har dessuten en package-manager som ofte gjør oppsett på tvers av
 operativsystemer enklere.
 
 Anaconda kan du finne og laste ned [her](https://www.anaconda.com/distribution/)
 
 ## Make for Windows
 
 Om du er på en windows-maskin og ønsker enkleste mulig oppsett anbefaler jeg
 å installere *make* på maskinen din. Hvordan du kan installere make for
 Windows kan du finne beskrevet [her](https://stackoverflow.com/a/54086635).
 
 **Om du ikke ønsker å installere make, kan du bare gå inn i ```Makefile```
 for å finne og kopiere de kommandoene som kjøres.**
 
 ## Installering av pakker
 
For å sette på Python-miljøet, kjør:
 ```
make env
```

Om du endrer environment.yml og legger til flere pakker, kjør:
```
make update-env
```

# Kjøring av presentasjonen

For å åpne presentasjonen, kjør:
```
make lab
```
Eller:
```
make notebook
```
Dette vil åpne et utviklingsverktøy kalt *Jupyter* i nettleseren din.

```lab``` er den nyeste versjonen av dette verktøyet, men har ikke enda
innebygget presentasjonsfunksjonalitet, noe ```notebook``` derimot har.

Naviger deretter til mappen *notebooks*, og åpne ```Presentation.ipynb```.

**Obs**: Filen er ganske stor, så det kan ta litt tid. Jeg vil sannsynligvis
splitte notebooken opp i flere notebooks senere.

# Nedlasting av filer brukt av presentasjonen

For å laste ned data fra diskusjon.no, som brukes gjennom hele presentasjonen, kjør:
```
make get-data
```

## Nedlasting av word embeddings

I presentasjonen snakker vi om noe som heter word embeddings. Jeg vil etterhvert sette opp
script for å laste ned disse, men per nå er du nødt til å laste dem ned selv.
Linker til de aktuelle filene, og hvor de skal lagres er tilgjengelig
der de brukes i presentasjonen.
 
 # Annen kode
 
 Repoet inneholder også mye kode som kan være nyttig for andre prosjekter. Disse finner du i mappen
 ```nlp_intro```. Stort sett alt som er her brukes i presentasjonen, så om du lurer hva en modul
 eller funksjon brukes til (og jeg ikke har dokumentert dette) - sjekk presentasjonen.