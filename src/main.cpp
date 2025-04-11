#include <Arduino.h>
#include <matrix.h>
#include <esp_heap_caps.h>
#include <value.h>

// Global variables to store model parameters
Value* W1_global = nullptr;
Value* W2_global = nullptr;

// Training parameters - reduced batch size for memory efficiency
const int num_points = 100;         // Reduced from 20 to 10
const float learning_rate = 0.01f;
const int max_epochs = 1000;
const int hidden_size = 128;        // Reduced hidden layer size
const float PI2 = 2.0f * PI;

// Helper function to allocate memory in PSRAM with fallback
void* allocateMemory(size_t size, bool prefer_psram = true) {
    void* ptr = nullptr;
    if (prefer_psram) {
        ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
    }
    if (!ptr) {
        ptr = heap_caps_malloc(size, MALLOC_CAP_8BIT);
    }
    if (!ptr) {
        Serial.printf("Failed to allocate %d bytes\n", size);
    }
    return ptr;
}

void printMemoryInfo() {
    Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
    if (psramFound()) {
        Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
    }
}

float** create_data_array(int rows, int cols, std::function<float(int, int)> init_func) {
    float** data = (float**)allocateMemory(rows * sizeof(float*), false); // Headers in regular memory
    if (!data) return nullptr;
    
    for (int i = 0; i < rows; i++) {
        data[i] = (float*)allocateMemory(cols * sizeof(float), true); // Data in PSRAM
        if (!data[i]) {
            for (int j = 0; j < i; j++) {
                heap_caps_free(data[j]);
            }
            heap_caps_free(data);
            return nullptr;
        }
        for (int j = 0; j < cols; j++) {
            data[i][j] = init_func(i, j);
        }
    }
    return data;
}

void free_data_array(float** data, int rows) {
    if (data) {
        for (int i = 0; i < rows; i++) {
            if (data[i]) heap_caps_free(data[i]);
        }
        heap_caps_free(data);
    }
}

float mmse(Value &y_true, Value &y_pred) {
    float result = 0.0f;
    if (y_true.ptr->rows != y_pred.ptr->rows || y_true.ptr->cols != y_pred.ptr->cols) {
        Serial.println("Error: shape mismatch");
        return result;
    }

    for (int i = 0; i < y_true.ptr->rows; i++) {
        for (int j = 0; j < y_true.ptr->cols; j++) {
            float diff = y_true.ptr->data[i][j] - y_pred.ptr->data[i][j];
            result += diff * diff;
        }
    }
    result /= static_cast<float>(y_true.ptr->rows * y_true.ptr->cols);

    if (y_pred.ptr->grad) {
        float scale = 2.0f / static_cast<float>(y_pred.ptr->rows * y_pred.ptr->cols);
        for (int i = 0; i < y_pred.ptr->rows; i++) {
            for (int j = 0; j < y_pred.ptr->cols; j++) {
                y_pred.ptr->grad[i][j] = scale * (y_pred.ptr->data[i][j] - y_true.ptr->data[i][j]);
            }
        }
    }
    return result;
}

Value* createTrainData(int points, bool is_x_data) {
    float** data = create_data_array(points, is_x_data ? 2 : 1, 
        [points, is_x_data](int i, int j) -> float {
            if (is_x_data) {
                if (j == 1) return 1.0f;
                return static_cast<float>(i) / static_cast<float>(points) * PI2;
            }
            return sin(static_cast<float>(i) / static_cast<float>(points) * PI2);
        });
    
    if (!data) return nullptr;
    
    Value* val = new Value(points, is_x_data ? 2 : 1, data, 
                          is_x_data ? "x_train" : "y_train");
    free_data_array(data, points);
    return val;
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    if (psramInit()) {
        Serial.println("PSRAM initialized");
        printMemoryInfo();
    } else {
        Serial.println("No PSRAM available");
    }

    // Create training data
    Serial.println("Creating training data...");
    Value* x_train = createTrainData(num_points, true);
    if (!x_train) {
        Serial.println("Failed to create x_train");
        return;
    }
    printMemoryInfo();

    Value* y_train = createTrainData(num_points, false);
    if (!y_train) {
        Serial.println("Failed to create y_train");
        delete x_train;
        return;
    }
    printMemoryInfo();

    // Create model parameters
    Serial.println("Creating model parameters...");
    float** w1_data = create_data_array(2, hidden_size, [](int i, int j) -> float {
        return random(-100, 100) / 100.0f;
    });
    if (!w1_data) {
        Serial.println("Failed to create W1");
        delete x_train;
        delete y_train;
        return;
    }
    W1_global = new Value(2, hidden_size, w1_data, "W1");
    free_data_array(w1_data, 2);
    printMemoryInfo();

    float** w2_data = create_data_array(hidden_size, 1, [](int i, int j) -> float {
        return random(-100, 100) / 100.0f;
    });
    if (!w2_data) {
        Serial.println("Failed to create W2");
        delete x_train;
        delete y_train;
        delete W1_global;
        return;
    }
    W2_global = new Value(hidden_size, 1, w2_data, "W2");
    free_data_array(w2_data, hidden_size);
    printMemoryInfo();

    // Training loop
    Serial.println("\nStarting training...");
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        Value hidden = (*x_train) * (*W1_global);
        Value hidden_act = hidden.leakyrelu();
        Value out = hidden_act * (*W2_global);

        float loss = mmse(*y_train, out);
        
        out.backward();
        W1_global->update(learning_rate);
        W2_global->update(learning_rate);
        
        W1_global->setgradzero();
        W2_global->setgradzero();
        out.setgradzero();

        if (epoch % 100 == 0) {
            Serial.printf("Epoch %d/%d: Loss = %.6f\n", epoch, max_epochs, loss);
            printMemoryInfo();
        }
        yield();
    }

    // Cleanup training data
    delete x_train;
    delete y_train;
    printMemoryInfo();

    Serial.println("\nModel trained! Enter a number to predict sin(x).");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        float x = input.toFloat();
        
        float** input_data = create_data_array(1, 2, [x](int i, int j) -> float {
            return j == 1 ? 1.0f : x;
        });
        
        if (input_data) {
            Value input_tensor(1, 2, input_data, "test_input");
            Value hidden = input_tensor * (*W1_global);
            Value hidden_act = hidden.leakyrelu();
            Value pred = hidden_act * (*W2_global);
            
            Serial.printf("sin(%.6f) ≈ %.6f\n", x, pred.ptr->data[0][0]);
            Serial.printf("Actual: %.6f\n", sin(x));
            
            free_data_array(input_data, 1);
        }
        
        while (Serial.available() > 0) {
            Serial.read();
        }
    }
    yield();
}
