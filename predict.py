import flask
import tf_utils as tfu

def get_response(predictions):
    votesForInTram = sum(predictions)
    isIntram = votesForInTram > (len(predictions) * 0.5)
    certainty = votesForInTram / len(predictions)
    return {
        'isInTram': True if isIntram else False,
        'certainty': certainty if isIntram else 1 - certainty,
    }

def predict_with(selected_predictor):
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json()
        if json != None:
            preds = []
            for data_row in json:
                model_input = tfu.get_example_from(data_row).SerializeToString()

                output_dict = selected_predictor({"inputs": [model_input]})
                outScore, inTramScore = output_dict['scores'][0]

                preds.append(inTramScore)

            data = get_response(preds)
            print(data)
            return flask.jsonify(data)
        else:
            data['error'] = 'no json'
    else:
        data['error'] = 'unsupported method'
    return flask.jsonify(data)