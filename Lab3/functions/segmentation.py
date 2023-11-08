class CellProcessor:
    def __init__(self):
        self.last5s_cell1 = RingBuffer(1500)
        self.last5s_cell2 = RingBuffer(1500)
        self.histBufferSize = 150
        self.historyBuffer_cell1 = RingBuffer(self.histBufferSize)
        self.historyBuffer_cell2 = RingBuffer(self.histBufferSize)
        self.sliceArray_cell1 = []
        self.sliceArray_cell2 = []
        self.busyReading_cell1 = False
        self.busyReading_cell2 = False
        self.avgMean_cell1 = 0
        self.avgMean_cell2 = 0
        self.historyMean_cell1 = 100000
        self.historyMean_cell2 = 100000
        self.samplingStep_cell1 = 0
        self.samplingStep_cell2 = 0
        self.slice_cell1_arr = [] # all the segmentation slices is stored in this list for one signal
        self.slice_cell2_arr = [] 
        self.test_cell1_arr = []
        self.test_cell2_arr = []
        # Segmentation multiplier value:
        self.thresholdMultiplier = 0.9
        self.defaultSliceSize = 600 - self.histBufferSize
        self.sliceSize = self.defaultSliceSize

    def split_training_test(self, trainingLength):
        self.test_cell1_arr = self.slice_cell1_arr[trainingLength:]
        self.slice_cell1_arr = self.slice_cell1_arr[0:trainingLength]
        self.test_cell2_arr = self.slice_cell2_arr[trainingLength:]
        self.slice_cell2_arr = self.slice_cell2_arr[0:trainingLength]

    def process_signal(self, list1, list2):
        for cell1_reading, cell2_reading in zip(list1, list2):
            if (not self.busyReading_cell1):
                self.last5s_cell1.append(cell1_reading)
                self.avgMean_cell1 = np.mean(self.last5s_cell1.get())
                self.historyBuffer_cell1.append(cell1_reading)
                self.historyMean_cell1 = np.mean(self.historyBuffer_cell1.get())
                
    
            if (not self.busyReading_cell2):
                self.last5s_cell2.append(cell2_reading)
                self.avgMean_cell2 = np.mean(self.last5s_cell2.get())
                self.historyBuffer_cell2.append(cell2_reading)
                self.historyMean_cell2 = np.mean(self.historyBuffer_cell2.get())
            # Check if an actual valid detection has occured
            detectChangeCell1 = (self.historyMean_cell1 < self.thresholdMultiplier * self.avgMean_cell1 and self.samplingStep_cell1 < self.sliceSize)
            detectChangeCell2 = (self.historyMean_cell2 < self.thresholdMultiplier * self.avgMean_cell2 and self.samplingStep_cell2 < self.sliceSize)
            if (detectChangeCell1 or detectChangeCell2):
                self.busyReading_cell1 = True
                self.busyReading_cell2 = True
                # Check if the sliceArray is empty. If so, it needs to be filled with the historyBuffer
                if not self.sliceArray_cell1:
                    self.sliceArray_cell1 = self.historyBuffer_cell1.get()
                else:
                    self.sliceArray_cell1.append(cell1_reading)
                    self.historyBuffer_cell1.append(cell1_reading)
                    
                if not self.sliceArray_cell2:
                    self.sliceArray_cell2 = self.historyBuffer_cell2.get()
                else:
                    self.sliceArray_cell2.append(cell2_reading)
                    self.historyBuffer_cell2.append(cell2_reading)
                
                self.samplingStep_cell1 += 1
                self.samplingStep_cell2 += 1
                self.gestureDetected = True
                self.gestureDetected_cell1 = True
                self.gestureDetected_cell2 = True
                
            elif (self.busyReading_cell1 or self.busyReading_cell2):
                self.historyMean_cell1 = np.mean(self.historyBuffer_cell1.get())
                self.historyMean_cell2 = np.mean(self.historyBuffer_cell2.get())
                isC1Done = (self.historyMean_cell1 < self.thresholdMultiplier * self.avgMean_cell1)
                isC2Done = (self.historyMean_cell2 < self.thresholdMultiplier * self.avgMean_cell2)
                if ( isC1Done or isC2Done ):
                    self.sliceSize += 300
                    self.historyBuffer_cell1.append(cell1_reading)
                    self.historyBuffer_cell2.append(cell2_reading)
                    continue
                    
                self.sliceArray_cell1.append(cell1_reading)
                self.sliceArray_cell2.append(cell2_reading)
#                 plt.plot(self.sliceArray_cell1, label="cell 1")
#                 plt.plot(self.sliceArray_cell2, label="cell 2")
#                 plt.show()
                print("Finished 1 slice")
                self.slice_cell1_arr.append(self.sliceArray_cell1)
                self.slice_cell2_arr.append(self.sliceArray_cell2)

                # reset
                self.samplingStep_cell1 = 0
                self.samplingStep_cell2 = 0
                self.busyReading_cell1 = False
                self.busyReading_cell2 = False
                self.historyMean_cell1 = 1000000
                self.historyMean_cell2 = 1000000
                self.historyBuffer_cell1.clear()
                self.historyBuffer_cell2.clear()
                self.sliceArray_cell1 = []
                self.sliceArray_cell2 = []
                self.sliceSize = self.defaultSliceSize
                
                
# Create the instance
processor_flat = CellProcessor()
processor_vertical = CellProcessor()
processor_flat_inverse = CellProcessor()
processor_vertical_inverse = CellProcessor()
processor_up = CellProcessor()
processor_down = CellProcessor()
processor_up_down = CellProcessor()
processor_down_up = CellProcessor()


# process the signal
processor_flat.process_signal(solar_cell_1_flat, solar_cell_2_flat)
processor_vertical.process_signal(solar_cell_1_vertical, solar_cell_2_vertical)
processor_up.process_signal(solar_cell_1_up, solar_cell_2_up)
processor_down.process_signal(solar_cell_1_down, solar_cell_2_down)
processor_up_down.process_signal(solar_cell_1_up_down, solar_cell_2_up_down)
processor_down_up.process_signal(solar_cell_1_down_up, solar_cell_2_down_up)
processor_flat_inverse.process_signal(solar_cell_1_flat_inverse, solar_cell_2_flat_inverse)
processor_vertical_inverse.process_signal(solar_cell_1_vertical_inverse, solar_cell_2_vertical_inverse)
               