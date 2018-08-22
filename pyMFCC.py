import pyaudio
import json
from sys import byteorder
from sys import version
from array import array
from scipy import fftpack
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy
import wave
import decimal


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

if version[0] == '3':
    xrange = range
if version[0] == '2':
    input = raw_input

class AudioSample(object):

    def __init__(self, frame_size = 0.025,
                sample_rate = 44100,
                frame_stride = 0.01,
                NFFT = 2048,
                num_filts = 40,
                num_ceps = 20,
                sample_array = None,
                sample_frames = None,
                fft_frames = None,
                abs_fft_frames = None,
                hamming_frames = None,
                power_spectrums = None,
                filter_bank = None,
                energy_total_per_frame = None,
                sample_energies = None,
                sample_log_energies = None,
                MFCCs = None):

        #These attributes have default values other than None
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.NFFT = NFFT
        self.num_filts = num_filts
        self.num_ceps = num_ceps

        #Other attributes have None as default but allow for initializing the
        #the object with already computed data points for skipping functions
        self.sample_array = sample_array
        self.sample_frames = sample_frames
        self.fft_frames = fft_frames
        self.abs_fft_frames = abs_fft_frames
        self.hamming_frames = hamming_frames
        self.power_spectrums = power_spectrums
        self.filter_bank = filter_bank
        self.energy_total_per_frame = energy_total_per_frame
        self.sample_energies = sample_energies
        self.sample_log_energies = sample_log_energies
        self.MFCCs = MFCCs

    def compute_MFCC(self):
        """
        Method: compute_MFCC

        Parameters: None

        Returns: MFCCs, the mel frequency cepstral coefficients of each frame in
                 the self.sample_array attribute. Default
                 number of MFCCs per frame is 20, can be changed during initialization
                 of the AudioSample object.

        Calls each function needed to compute the MFFCs of the audio data. The
        self.sample_array must be initialized either through the record()
        method or the load_audio() method, else raises an MFCC_CALC_EXC.

        """

        if isinstance(self.sample_array, type(None)):
            raise MFCC_CALC_EXC("No audio samples loaded.")

        self.split_audio()
        self.hamming_windows()
        self.transform_frames()
        self.power_spectrum_frames()
        self.compute_filterbanks()
        self.compute_energies()
        self.compute_log()
        DCTs = self.compute_DCT()

        #keep only the first num_ceps MFCCs
        self.MFCCs = DCTs[:,:self.num_ceps]
        return self.MFCCs

    def compute_DCT(self):
        """
        Method: compute_DCT

        Parameters: None

        Returns: dcts, the mels frequency cepstral coefficients of each frame.
                 Default number of mfccs per frame is 40, based on the 40 filter
                 banks.

        Computes the discrete cosine transform of each log-energy of each frame.
        From the scipy.fftpack doc:
                       N-1
             y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                       n=0
        When norm="ortho", y is multiplied by scaling factor f:
            f = sqrt(1/(4*N)) if k = 0,
            f = sqrt(1/(2*N)) otherwise.
        """
        dcts = fftpack.dct(self.sample_log_energies, type = 2, axis = 1, norm = 'ortho')
        return dcts

    def compute_log(self):
        """
        Method: compute_log

        Parameters: None

        Returns: sample_log_energies, the log of each energy in each frame.

        Computes the log of each energy in each frame of the sample. Raises MFCC_CALC_EXC
        if the sample_energies has not been calculated yet.

        """
        if isinstance(self.sample_energies, type(None)):
            raise MFCC_CALC_EXC("Energies not computed.")

        self.sample_log_energies = numpy.log(self.sample_energies)
        return self.sample_log_energies

    def compute_energies(self):
        """
        Method: compute_energies

        Parameters: None

        Returns: energy_total_per_frame, sample_energies

        Computes the energy total per frame, which is the summation of each value
        in the power spectrum of the frame. Then computes the sample energies
        by passing the power spectrum of each frame through the filter bank. If
        one of the sample energies is 0, replaces with a small number so that
        there are no problems with the log in the next calculation.
        """
        if isinstance(self.filter_bank, type(None)) or isinstance(self.power_spectrums, type(None)):
            raise MFCC_CALC_EXC("Either filter bank or power spectrum not computed.")

        self.energy_total_per_frame = numpy.sum(self.power_spectrums, 1)

        #energies of each frame are calculated by taking the dot product of the
        #power spectrum and the transpose of the filter bank
        self.sample_energies = numpy.dot(self.power_spectrums, self.filter_bank.T)

        #replace 0 values
        self.sample_energies = numpy.where(self.sample_energies == 0,numpy.finfo(float).eps,self.sample_energies)

        return self.energy_total_per_frame, self.sample_energies

    def compute_filterbanks(self):

        """
        Method: compute_filterbank

        Parameters: None

        Returns: filter_bank

        Computes the filter bank, an array that seperates the input signal into
        frequency components. Must be spaced according to the mel scale, which is
        then converted to the Hertz scale. The filters in the bank are triangular,
        and follow the following equation:

                    0                                  if k < f(m-1)
                    [k - f(m-1)]/[f(m) - f(m-1)]       if f(m -1) <= k < f(m)
        Hm(k) = {   1                                  if k = f(m)
                    [f(m + 1) - k]/[f(m + 1) - f(m)]   if f(m) < k <= f(m + 1)
                    0                                  if k > f(m - 1)
        """

        #The lowest point in the filter bank
        mel_low = 0

        #The highest point in the filter bank. Converts half the sample rate
        #of the Audio Sample to mels.
        mel_up = (2595 * numpy.log10(1 + (self.sample_rate/2) / 700))

        melpoints = numpy.linspace(mel_low,mel_up,self.num_filts+2)

        #Converts the linearly-spaced mels to the Hertz scale
        hz_pts = (700 * (10 ** (melpoints/2595.0) - 1))

        bin = numpy.floor((self.NFFT+1)*hz_pts/self.sample_rate)


        fbank = numpy.zeros([self.num_filts,self.NFFT//2+1])

        #This for loop follows the Hm(k) equation documented above
        for j in range(0,self.num_filts):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])


        self.filter_bank = fbank

        return self.filter_bank

    def power_spectrum_frames(self):
        """

        Method: power_spectrum_frames

        Parameters: None

        Returns: power_spectrums

        Computes the power spectrum of each frame.

        """
        if isinstance(self.fft_frames, type(None)):
            raise MFCC_CALC_EXC("The fft formula has not been applied")

        self.power_spectrums = 1.0/self.NFFT * numpy.square(self.abs_fft_frames)

        self.power_spectrums = numpy.array(self.power_spectrums)
        return self.power_spectrums

    def hamming_windows(self):

        """
        Method: hamming_windows

        Parameters: None

        Returns: hamming_frames

        Computes a hamming window according to the frame size. Applies the hamming
        window to each frame.
        """
        if isinstance(self.sample_frames, type(None)):
            raise MFCC_CALC_EXC("No sample frames")

        self.hamming_frames = numpy.array(self.sample_frames)
        hamming_window = numpy.hamming(self.frame_length)
        self.hamming_frames = hamming_window * self.hamming_frames
        return self.hamming_frames

    def transform_frames(self):
        """
        Method: transform_frames

        Parameters: None

        Returns: fft_frames

        Computes the Fast Fourier Transform of each frame. To produce a proper FFT
        a window function must be applied. The hamming window function is used.
        Then computes the absolute value of each value in each frame, for use
        in the power spectrum calculation. NFFT default value is 2048 (the next
        highest power of 2 after the default frame length of [frame_rate * frame_size],
        which represents a frame length of 0.025 seconds)

        """
        if isinstance(self.hamming_frames, type(None)):
            raise MFCC_CALC_EXC("The hamming window formula has not been applied")


        self.fft_frames = numpy.fft.rfft(self.hamming_frames, self.NFFT)

        #calculate the absolute value of the fourier transformed frames, for use
        #in the calculation of the power spectrum
        self.abs_fft_frames = numpy.absolute(self.fft_frames)

        return self.fft_frames

    def split_audio(self):

        """
        Method: split_audio

        Parameters: None

        Returns: sample_frames

        Splits the audio sample into overlapping subframes. Calculates the length
        and step using the time values (frame_size * sample_rate) and
        (frame_step * sample_rate) respectively.
        """

        #Round up for the frame length using the precision values obtained through the decimal module
        self.frame_length = int(decimal.Decimal(self.frame_size * self.sample_rate).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        curr_frame_end = self.frame_length
        self.frame_step = int(decimal.Decimal(self.frame_stride * self.sample_rate).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        curr_frame_front = 0

        sub_lists = []
        while(True):
            #If the current point to start on and end on is not past the end of the array,
            #cut a chunk out of the sample array
            if curr_frame_end < len(self.sample_array) - 1 and curr_frame_front < len(self.sample_array) - 1:
                sub_list = self.sample_array[curr_frame_front:curr_frame_end]
                sub_lists.append(sub_list)

                #take a step in the forward direction
                curr_frame_end = curr_frame_end + self.frame_step
                curr_frame_front = curr_frame_front + self.frame_step

            #if the end of the current frame goes past the last element but the frame
            #beginning is in front of the last element, cut the rest of the signal off
            #into the final sublist
            elif curr_frame_end > len(self.sample_array) - 1 and curr_frame_front <= len(self.sample_array) - 1:
                sub_list = self.sample_array[curr_frame_front:]
                sub_lists.append(sub_list)
                break
            #if the current frame ends at the last point in the sample array,
            #do the same as above
            elif curr_frame_end == len(self.sample_array) - 1:
                sub_list = self.sample_array[curr_frame_front:]
                sub_lists.append(sub_list)
                break
            #if the current frame front and the current frame end go past the end
            #break the loop
            else:
                break

        #zero pad up to the frame size the last sub_list
        last_list = sub_lists[-1]
        if len(last_list) < self.frame_length:
            while len(last_list) < self.frame_length:
                last_list.append(0)

        self.sample_frames = sub_lists
        return self.sample_frames

    def save_data(self, save_array):
        """
        Method: save_data

        Parameters: save_array, a numpy ndarray to be saved

        Returns: None

        Saves the numpy array as a list in a json file. Usage note: type the name of
        the file to be saved without the extension.
        """

        print("Enter filename to save as: ")
        filename = input()

        save_list = save_array.tolist()

        with open(filename + ".json", 'w') as f:
            json.dump(save_list, f)

    def load_audio(self, fname):
        """
        Method: load_audio

        Parameters: fname, the wav file to be read from.

        Return: graph(self.sample_array), a pyplot representation of the
                sample array over time

        Usage notes: Must include .wav extension in fname parameter string.
        Use [return val].show() to view the sample array. Only usable as a
        """
        sample_rate, stereo_array = wavfile.read(fname)
        sample_array = stereo_array.astype(float)

        #for stereo audio, average the two channels together
        if(len(sample_array.shape) == 2):
            sample_array = stereo_array.sum(axis=1) // 2
        sample_array = sample_array.tolist()
        self.sample_rate = sample_rate
        self.sample_array = sample_array

        return graph(self.sample_array)

    def normalize(self, snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

    def trim(self, snd_data):
        "Trim the blank spots at the start and end"
        def _trim(snd_data):
            snd_started = False
            r = array('h')

            for i in snd_data:
                if not snd_started and abs(i)>THRESHOLD:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def record(self):
        """
        Record a word or words from the microphone and
        return the data as an array of signed shorts.

        Normalizes the audio, and trims silence from the
        start and end, and returns the sample width, time, and data.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE,
            input=True, output=True,
            frames_per_buffer=CHUNK_SIZE)

        num_silent = 0
        snd_started = False

        r = array('h')

        while 1:
            # little endian, signed short
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            silent = False
            if max(snd_data) < THRESHOLD:
                silent = True

            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > 30:
                break

        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = self.normalize(r)
        r = self.trim(r)
        self.sample_array = r
        return sample_width, self.sample_array



def graph_fft(*pargs):

    if len(pargs) == 1:
        r = []

        for item in pargs[0]:
            mag = mag_of_complex(item)
            r.append(mag)
        freq_spacing = 44100/len(pargs[0])
        freq_list = []

        for i in range(int(len(r)/2)):
            freq_list.append(freq_spacing * i)

        plt.plot(freq_list, r[:len(freq_list)])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|Y(f)| (Mag of FT Coeff)")
        plt.tight_layout()

        return plt

    else:
        fig, ax = plt.subplots(nrows = len(pargs), ncols = 1)
        counter = 0

        for x in pargs:
            r = []

            for item in x:
                mag = mag_of_complex(item)
                r.append(mag)

            freq_spacing = 44100/len(x)
            freq_list = []

            for i in range(int(len(r)/2)):
                freq_list.append(freq_spacing * i)

            ax[counter].plot(freq_list, r[:len(freq_list)])
            ax[counter].set_xlabel("Frequency (Hz)")
            ax[counter].set_ylabel("|Y(f)| (Mag of FT Coeff)")
            counter += 1

        plt.tight_layout()
        return plt




def mag_of_complex(x):
    mag = numpy.sqrt(x.real**2 + x.imag**2)
    return mag

def graph(*pargs):

    if(len(pargs) == 1):
        interval = 0.0
        time_vec = []

        for i in range(len(pargs[0])):
            time_vec.append(interval)
            interval += (len(pargs[0])/RATE)/len(pargs[0])

        plt.plot(time_vec, pargs[0])
        plt.xlabel("Time")
        plt.ylabel("Sample Value")
        plt.tight_layout()
        return plt


    fig, ax = plt.subplots(nrows = len(pargs), ncols = 1)
    counter = 0

    for y in pargs:
        interval = 0.0
        time_vec = []
        for i in range(len(y)):
            time_vec.append(interval)
            interval += (len(y)/RATE)/len(y)

        ax[counter].plot(time_vec, y)
        ax[counter].set_xlabel("Time")
        ax[counter].set_ylabel("Sample Value")
        counter+=1

    plt.tight_layout()
    return plt

def graph_from_list(l):

    fig, ax = plt.subplots(nrows = len(l), ncols = 1)
    counter = 0

    for y in l:
        interval = 0.0
        time_vec = []
        for i in range(len(y)):
            time_vec.append(interval)
            interval += (len(y)/RATE)/len(y)

        ax[counter].plot(time_vec, y)
        ax[counter].set_xlabel("Time")
        ax[counter].set_ylabel("Sample Value")
        counter+=1

    plt.tight_layout()
    return plt

class MFCC_CALC_EXC(Exception):
    pass
