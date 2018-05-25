import tensorflow as tf


COLUMNS_COUNT = 1002

LABELS = ['almondjoy ', 'bounty ', 'dove ', 'mars ', 'milkyway ', 'snickers ',
          'threemusketeers ', 'twix ']

LABEL_COLUMN = 'candy_name'

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'

CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'

def model_fn(mode,
             features,
             labels,
             embedding_size=8,
             learning_rate=0.1):
  """Create a Feed forward network classification network

  Args:
    mode (string): Mode running training, evaluation or prediction
    features (dict): Dictionary of input feature Tensors
    labels (Tensor): Class label Tensor
    learning_rate (float): Learning rate for the SGD

  Returns:
    Depending on the mode returns Tuple or Dict
  """
  

  # Concatenate the (now all dense) features.
  # We need to sort the tensors so that they end up in the same order for
  # prediction, evaluation, and training
  for col in features:
    # Give continuous columns an extra trivial dimension
    # So they can be concatenated with embedding tensors
    features[col] = tf.expand_dims(tf.to_float(features[col]), -1)
  sorted_feature_tensors = zip(*sorted(features.iteritems()))[1]
  inputs = tf.concat(sorted_feature_tensors, 1)

  # Add the output layer
  logits = tf.layers.dense(
    inputs,
    len(LABELS),
    activation=None,
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
  )

  label_values = tf.constant(LABELS)
  tf.Print(label_values, [label_values], message="These are the label values: ")

  if mode in (PREDICT, EVAL):
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)

  if mode in (TRAIN, EVAL):
    # Convert the string label column to indices
    # Build a lookup table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(label_values)

    # Use the lookup table to convert string labels to ints
    label_indices = table.lookup(labels)

    # Make labels a vector
    label_indices_vector = tf.squeeze(label_indices)

    # global_step is necessary in eval to correctly load the step
    # of the checkpoint we are evaluating
    global_step = tf.contrib.framework.get_or_create_global_step()

  if mode == PREDICT:
    # Convert predicted_indices back into strings
    return {
        'predictions': tf.gather(label_values, predicted_indices),
        'confidence': tf.reduce_max(probabilities, axis=1)
    }

  if mode == TRAIN:
    # Build training operation.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=label_indices_vector)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cross_entropy)
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                       l1_regularization_strength=3.0,
                                      l2_regularization_strength=10.0)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)
    return train_op, global_step

  if mode == EVAL:
    # Return accuracy and area under ROC curve metrics
    # See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    # See https://www.kaggle.com/wiki/AreaUnderCurve\
    labels_one_hot = tf.one_hot(
        label_indices_vector,
        depth=label_values.shape[0],
        on_value=True,
        off_value=False,
        dtype=tf.bool
    )
    return {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
        'auroc': tf.metrics.auc(labels_one_hot, probabilities)
    }


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  CSV_COLUMNS = []
  CSV_COLUMN_DEFAULTS = []
  CONTINUOUS_COLS = []
  for i in range(0, COLUMNS_COUNT - 1):
    CSV_COLUMNS.append('feature_{}'.format(i))
    CONTINUOUS_COLS.append('feature_{}'.format(i))
    CSV_COLUMN_DEFAULTS.append([0.0])
  CSV_COLUMNS.append(LABEL_COLUMN)
  CSV_COLUMN_DEFAULTS.append([''])
  UNUSED_COLUMNS = set(CSV_COLUMNS) - set(CONTINUOUS_COLS + [LABEL_COLUMN])

  #columns = tf.decode_csv(rows_string_tensor, record_defaults=CSV_COLUMN_DEFAULTS)
  columns = tf.decode_csv(rows_string_tensor, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
  """Generates features and labels for training or evaluation.
  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

  Args:
      filenames (str): List of CSV files to read data from.
      num_epochs (int): How many times through to read the data.
                        If None will loop through data indefinitely
      shuffle (bool): Whether or not to randomize the order of data.
                      Controls randomization of both file order and line
                      order within files.
      skip_header_lines (int): Set to non-zero in order to skip header lines
                               in CSV files.
      batch_size (int): First dimension size of the Tensors returned by
                        input_fn.
  Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """

  dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(parse_csv)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features, features.pop(LABEL_COLUMN)
