from flask import Flask, request, jsonify
from flask_cors import CORS
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.protobuf import music_pb2
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Load a pretrained MelodyRNN model
bundle = sequence_generator_bundle.read_bundle_file(
    tf.keras.utils.get_file(
        'basic_rnn.mag', 
        'http://download.magenta.tensorflow.org/models/melody_rnn/basic_rnn.mag'
    )
)
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

def midi_json_to_note_sequence(midi_json):
    seq = music_pb2.NoteSequence()
    for note in midi_json:
        seq.notes.add(
            pitch=int(note['midiNumber']),
            start_time=note['timestamp']/1000.0,
            end_time=(note['timestamp'] + note['durationMs'])/1000.0,
            velocity=int(note['velocity']*127)
        )
    seq.total_time = max((note['timestamp'] + note['durationMs'])/1000.0 for note in midi_json)
    return seq

def note_sequence_to_json(seq):
    result = []
    for note in seq.notes:
        result.append({
            'noteName': note_seq.note_number_to_name(note.pitch),
            'midiNumber': note.pitch,
            'startTime': note.start_time,
            'endTime': note.end_time,
            'velocity': note.velocity/127
        })
    return result

@app.route('/generate', methods=['POST'])
def generate():
    try:
        midi_json = request.json
        input_seq = midi_json_to_note_sequence(midi_json)

        # Generate 4-second continuation (~1 measure at 120 BPM)
        generator_options = melody_rnn_sequence_generator.SequenceGeneratorOptions()
        generator_options.args['temperature'] = 1.0
        generator_options.generate_sections.add(
            start_time=input_seq.total_time,
            end_time=input_seq.total_time + 4.0
        )

        generated_seq = melody_rnn.generate(input_seq, generator_options)
        return jsonify(note_sequence_to_json(generated_seq))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
