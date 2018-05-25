import cStringIO
import unittest
import tensorflow as tf
import trainer.model as model


def Test(case):
  suite = unittest.TestLoader().loadTestsFromTestCase(case)
  stream = cStringIO.StringIO()
  unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
  print stream.getvalue()


class TestInputFn(tf.test.TestCase):
  def testBasic(self):
    with self.test_session() as session:
      features, indices = model.input_fn('gs://intelligent-candy-image-classifier/bottlenecks/training.csv')
      #features, indices = model.input_fn('gs://cloud-samples-data/ml-engine/census/data/adult.test.csv')
      session.run(features)
      tf.logging.info(features)
      session.run(indices)
      tf.logging.info(indices)


tf.logging.set_verbosity(tf.logging.DEBUG)
Test(TestInputFn)