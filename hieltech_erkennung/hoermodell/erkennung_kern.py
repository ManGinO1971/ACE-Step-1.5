"""Kern der neuen Instrumenten-Erkennung (EfficientAT mn40_as_ext, AudioSet) – Zuordnung zu den 53 App-Instrumenten."""
import numpy as np, csv, os
HIER=os.path.dirname(os.path.abspath(__file__))
# AudioSet-Klasse -> App-Instrument (53er-Slugs)
ZUORDNUNG={
 'guitar':['Guitar','Plucked string instrument','Strum','Tapping (guitar technique)','Ukulele','Mandolin'],
 'electric-guitar':['Electric guitar'], 'acoustic-guitar':['Acoustic guitar'],
 'banjo':['Banjo'], 'sitar':['Sitar'], 'dobro':['Steel guitar, slide guitar'], 'harp':['Harp'],
 'bass':['Bass guitar'], 'double-bass':['Double bass'],
 'piano':['Piano'], 'digital-piano':['Electric piano'], 'keys':['Keyboard (musical)'], 'harpsichord':['Harpsichord'],
 'organ':['Organ','Hammond organ','Electronic organ'],
 'synth':['Synthesizer','Sampler'],
 'saxophone':['Saxophone'], 'trumpet':['Trumpet'], 'trombone':['Trombone'], 'french-horn':['French horn'],
 'brass':['Brass instrument'], 'clarinet':['Clarinet'], 'flute':['Flute'], 'woodwind':['Wind instrument, woodwind instrument'],
 'violin':['Violin, fiddle'], 'cello':['Cello'], 'strings':['String section','Orchestra'], 'bowed_strings':['Bowed string instrument'],
 'harmonica':['Harmonica'], 'accordion':['Accordion'],
 'drums':['Drum kit','Drum machine'], 'kick':['Bass drum'], 'snare':['Snare drum','Rimshot','Drum roll'], 'hh':['Hi-hat','Cymbal'],
 'percussion':['Percussion','Drum','Tabla','Wood block','Maraca','Rattle (instrument)','Cowbell','Clapping','Gong','Steelpan'],
 'tambourine':['Tambourine'], 'timpani':['Timpani'],
 'marimba':['Marimba, xylophone','Mallet percussion'], 'glockenspiel':['Glockenspiel','Vibraphone'], 'bells':['Bell','Tubular bells','Church bell','Jingle bell'],
 'wind-chimes':['Wind chime','Chime'],
 'vocal':['Singing','Vocal music','Male singing','Female singing','Rapping','Humming','A capella'], 'back-vocal':['Choir','Chant'],
}
def labels(eat_dir):
    return [r[2] for r in list(csv.reader(open(os.path.join(eat_dir,'metadata','class_labels_indices.csv'))))[1:]]
def slug_scores(P,lab):
    L={l:i for i,l in enumerate(lab)}
    return {s:P[:,[L[x] for x in ks if x in L]].max(1) for s,ks in ZUORDNUNG.items()}
