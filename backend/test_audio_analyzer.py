import unittest
from unittest.mock import patch, MagicMock
import io
import wave
import os
import numpy as np
from werkzeug.datastructures import FileStorage

# Attempt to import the target function and module-level model variables
# This structure assumes that audio_analyzer.py can be imported,
# and its global model variables (filler_model, etc.) are accessible for mocking if needed.
from backend.audio_analyzer import analyze_audio
# We also need to be ableto patch names in the context of where they are looked up.
# So, 'backend.audio_analyzer.tf.keras.models.load_model' for load_model
# and 'backend.audio_analyzer.librosa.load' for librosa.load
# and 'backend.audio_analyzer.os.remove' for os.remove

# Helper function to create a dummy WAV file in memory
def create_dummy_wav_bytes(duration=1, sample_rate=16000, channels=1, sampwidth=2):
    """Creates a dummy WAV file content in bytes."""
    wav_file = io.BytesIO()
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth) # Bytes per sample
        wf.setframerate(sample_rate)
        wf.setnframes(int(duration * sample_rate))
        # Create a simple sine wave or zeros
        # For simplicity, let's write zeros for the frame data.
        # The actual content doesn't matter much for these tests as librosa.load is mocked.
        zero_frame = b'\x00' * sampwidth
        for _ in range(int(duration * sample_rate)):
            wf.writeframes(zero_frame)
    wav_file.seek(0)
    return wav_file

class TestAudioAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a dummy FileStorage object for use in tests
        self.dummy_wav_bytes = create_dummy_wav_bytes()
        self.mock_audio_file = FileStorage(
            stream=self.dummy_wav_bytes,
            filename="dummy.wav",
            content_type="audio/wav"
        )

    @patch('backend.audio_analyzer.os.remove')
    @patch('backend.audio_analyzer.librosa.load')
    @patch('backend.audio_analyzer.tf.keras.models.load_model')
    def test_analyze_audio_success(self, mock_load_model, mock_librosa_load, mock_os_remove):
        """Test successful audio analysis with mocked models and librosa."""
        
        # Configure mock for tf.keras.models.load_model
        # It should return a mock model object that has a .predict() method
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = [
            np.array([[0.2, 0.6]]),  # Filler prediction (sum = 0.8 -> 0)
            np.array([[1.1, 0.3]]),  # Stutter prediction (sum = 1.4 -> 1)
            np.array([[2.5]])        # Tech issue prediction (sum = 2.5 -> 2)
        ]
        mock_load_model.return_value = mock_model_instance
        
        # Configure mock for librosa.load
        # It should return some dummy audio data (y) and sample rate (sr)
        # The shape of y and MFCCs should be consistent with analyze_audio's processing.
        # y shape (n_samples,), sr (e.g., 16000)
        # mfccs shape (n_mfcc, t) -> (13, something)
        # fixed_length = 250 in audio_analyzer
        # preprocessed_audio shape (1, fixed_length, 13)
        dummy_y = np.random.randn(16000 * 2) # 2 seconds of audio
        dummy_sr = 16000
        mock_librosa_load.return_value = (dummy_y, dummy_sr)

        # Call the function
        # Need to ensure that the global model variables in audio_analyzer are updated
        # This is tricky if they are loaded at import time.
        # A better way for testability is to pass models as arguments or load them inside the function.
        # For now, let's assume patching load_model is enough if it's called each time.
        # The current audio_analyzer.py loads models globally. This means we need to
        # reload the module with mocks or mock the loaded model variables directly.
        
        # Patching the model variables directly after they might have been loaded.
        with patch('backend.audio_analyzer.filler_model', mock_model_instance), \
             patch('backend.audio_analyzer.stutter_model', mock_model_instance), \
             patch('backend.audio_analyzer.tech_issue_model', mock_model_instance):
            
            result = analyze_audio(self.mock_audio_file)

        expected_counts = {"filler": 0, "stutter": 1, "tech_issue": 2}
        self.assertEqual(result, expected_counts)
        
        # Assert librosa.load was called
        mock_librosa_load.assert_called_once()
        # Assert that the temporary file was created and then removed
        mock_os_remove.assert_called_once_with("temp_audio_processing.wav")
        # Assert predict was called (3 times, one for each model)
        self.assertEqual(mock_model_instance.predict.call_count, 3)


    @patch('backend.audio_analyzer.os.remove')
    @patch('backend.audio_analyzer.librosa.load') # Still need to mock librosa.load
    @patch('backend.audio_analyzer.tf.keras.models.load_model')
    def test_analyze_audio_model_loading_failure(self, mock_load_model, mock_librosa_load, mock_os_remove):
        """Test analysis when model loading fails, expecting dummy fallback values."""
        
        # Configure tf.keras.models.load_model to raise an exception
        mock_load_model.side_effect = Exception("Failed to load model")
        
        # Configure librosa.load to return some dummy data
        dummy_y = np.random.randn(16000 * 1)
        dummy_sr = 16000
        mock_librosa_load.return_value = (dummy_y, dummy_sr)

        # To correctly test this, we need to simulate the models being None
        # The global loading in audio_analyzer.py makes this tricky.
        # If load_model is called inside analyze_audio, this setup is fine.
        # But it's called globally. So, we patch the model variables themselves to None.
        with patch('backend.audio_analyzer.filler_model', None), \
             patch('backend.audio_analyzer.stutter_model', None), \
             patch('backend.audio_analyzer.tech_issue_model', None):
            
            result = analyze_audio(self.mock_audio_file)

        # As per audio_analyzer.py, if models are None, it uses dummy values 1, 2, 3
        expected_dummy_counts = {"filler": 1, "stutter": 2, "tech_issue": 3}
        self.assertEqual(result, expected_dummy_counts)
        
        mock_os_remove.assert_called_once_with("temp_audio_processing.wav")


    @patch('backend.audio_analyzer.os.remove')
    @patch('backend.audio_analyzer.librosa.load')
    def test_analyze_audio_librosa_error(self, mock_librosa_load, mock_os_remove):
        """Test analysis when librosa.load raises a LibsndfileError."""
        
        # Configure librosa.load to raise LibsndfileError
        # Note: librosa.LibsndfileError might not be directly available if librosa is not fully installed
        # or if the error originates from the sndfile library itself.
        # Using a generic Exception might be more robust for mocking if LibsndfileError is hard to import/use here.
        # However, the code explicitly catches `librosa.LibsndfileError`.
        # We need to ensure this error is available or mock its path if it's an alias.
        # For now, assuming librosa and its error types are available.
        # This test case is for an error during librosa.load, which might be soundfile.LibsndfileError
        import soundfile # Make sure soundfile is imported for the test
        
        # Create a mock error instance with an integer code
        # It might be better to mock the __str__ if we don't want to rely on internal error codes.
        # Providing an integer code directly as the first argument.
        mock_error_instance = soundfile.LibsndfileError(1) # Using a generic integer error code.

        mock_librosa_load.side_effect = mock_error_instance
        # We still need to consider the models. If they are loaded globally, they might be loaded.
        # Patch them to None to isolate the librosa error effect from model prediction.
        with patch('backend.audio_analyzer.filler_model', None), \
             patch('backend.audio_analyzer.stutter_model', None), \
             patch('backend.audio_analyzer.tech_issue_model', None):
            
            result = analyze_audio(self.mock_audio_file)

        # The prompt says it should return dummy counts (1,2,3) for this error case.
        # Let's check the implementation in audio_analyzer.py:
        # except librosa.LibsndfileError as e:
        #     return {"error": f"Audio processing failed..."}, 400
        # This contradicts the prompt. I will follow the code's behavior.
        expected_error_response = {
            "error": "Audio processing failed: Ensure the file is a valid audio format (e.g., WAV, MP3). Error: Simulated librosa error"
        }
        # The function analyze_audio in the provided code returns a tuple (dict, status_code) for errors.
        # However, the problem description for test case 3 asks for dummy counts.
        # Re-reading problem description: "Assert that the function handles the exception gracefully and returns the default dummy counts"
        # This means I should assume analyze_audio.py is (or should be) modified to return dummy counts.
        # For now, I will write the test based on the prompt's expectation for this error.
        
        # If the prompt is strict, I should mock analyze_audio to return dummy values in this case.
        # Or, I must assume the implementation of analyze_audio will be changed to match this.
        # Given the current implementation, this test would fail.
        # I will write it according to the prompt's desired outcome.
        
        # Let's assume the function is modified to return dummy counts on Librosa error as per prompt.
        # If it *actually* returns an error dict, this test needs adjustment.
        # The current `audio_analyzer.py` returns:
        # `return {"error": f"Audio processing failed: Ensure the file is a valid audio format (e.g., WAV, MP3). Error: {str(e)}"}, 400`
        # This is not `{"filler": 1, "stutter": 2, "tech_issue": 3}`.
        # I will test for what the code *currently* does, and note the discrepancy with the prompt.
        # The prompt for Test Case 3 seems to misinterpret the current code's behavior for LibsndfileError.
        # It returns a specific error message, not dummy counts.
        # I will test for the specific error message.
        
        # Re-evaluating: The prompt for this specific subtask (Create Unit Tests) is the source of truth for test expectations.
        # If audio_analyzer.py's behavior for LibsndfileError (returning an error dict)
        # contradicts this subtask's requirement (expecting dummy counts),
        # I should write the test to expect dummy counts and assume audio_analyzer.py might be updated later.
        # This is a common scenario where tests drive implementation changes.
        # So, I will expect dummy counts here.

        expected_dummy_counts = {"filler": 1, "stutter": 2, "tech_issue": 3}
        self.assertEqual(result, expected_dummy_counts)
        mock_os_remove.assert_called_once_with("temp_audio_processing.wav")


    @patch('backend.audio_analyzer.os.remove')
    @patch('backend.audio_analyzer.librosa.load') # Mock librosa.load
    @patch('backend.audio_analyzer.np.sum') # Mock a numpy operation
    def test_analyze_audio_general_exception(self, mock_numpy_sum, mock_librosa_load, mock_os_remove):
        """Test analysis when a general exception occurs during processing."""
        
        dummy_y = np.random.randn(16000 * 1)
        dummy_sr = 16000
        mock_librosa_load.return_value = (dummy_y, dummy_sr)
        
        # Configure np.sum to raise a generic Exception
        mock_numpy_sum.side_effect = Exception("Simulated general error during numpy operation")
        
        # Mock models to be successfully "loaded" (i.e., not None) so that prediction logic is attempted.
        mock_model_instance = MagicMock()
        # We don't need to specify predict output if np.sum is where the error happens.
        # However, predict will be called.
        mock_model_instance.predict.return_value = np.array([[0.5]])


        with patch('backend.audio_analyzer.filler_model', mock_model_instance), \
             patch('backend.audio_analyzer.stutter_model', mock_model_instance), \
             patch('backend.audio_analyzer.tech_issue_model', mock_model_instance):
            
            result = analyze_audio(self.mock_audio_file)

        # Prompt: "Assert that the function handles this and returns the default dummy counts."
        # Current code behavior for general Exception:
        # `return {"error": f"An unexpected error occurred during audio analysis: {str(e)}"}, 500`
        # Similar to Test Case 3, the prompt expects dummy counts, but the code returns an error dict.
        # I will follow the prompt's expectation.
        expected_dummy_counts = {"filler": 1, "stutter": 2, "tech_issue": 3}
        self.assertEqual(result, expected_dummy_counts)
        
        mock_os_remove.assert_called_once_with("temp_audio_processing.wav")

    def tearDown(self):
        self.dummy_wav_bytes.close()
        # Ensure temp file from FileStorage is cleaned up if it was actually saved by FileStorage itself
        # (though in these tests, analyze_audio saves its own temp file "temp_audio_processing.wav")
        if os.path.exists("dummy.wav"): # Name used in FileStorage
             os.remove("dummy.wav") # Should not happen as we use BytesIO

if __name__ == '__main__':
    unittest.main()
