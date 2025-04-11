#pragma once

#include <Arduino.h>
#include "matrix.h"
#include <cstring>
#include <cmath>
#include "minimal_intrusive_ptr.hpp"

class Value
{
public:
    mutable minimal::intrusive_ptr<Tensor> ptr;
    minimal::intrusive_ptr<Tensor> orig;

public:
    Value()
    {
        ptr = nullptr;
        orig = nullptr;
    };

    explicit Value(Tensor *t)
    {
        ptr = t;
    }

    Value(int row, int cols, float **data, std::string name)
    {
        ptr = minimal::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
        orig = ptr;
    }

    Value &operator=(const Value &other)
    {
        if (this != &other)
        {
            ptr = other.ptr;
        }
        return *this;
    }

    Value operator+(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr + *other.ptr));
    }

    Value operator*(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr * *other.ptr));
    }

    Value operator^(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr ^ *other.ptr));
    }

    Value operator/(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr / *other.ptr));
    }

    Value operator-(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr - *other.ptr));
    }

    Value leakyrelu(float leaky = 0.01)
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(ptr->lekyrelu(leaky)));
    }

    void setgrad(float **grad)
    {
        ptr->setGrad(grad);
    }

    void setgradzero()
    {
        if(this->ptr) {ptr->setgradzero();}
        if(this->orig) {orig->setgradzero();}
    }

    void backward()
    {
        ptr->backward();
    }

    void printgrad()
    {
        if (orig == nullptr)
        {
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    Serial.print(ptr->grad[i][j]);
                    Serial.print(" ");
                }
                Serial.println();
            }
        }
        else
        {
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    Serial.print(orig->grad[i][j]);
                    Serial.print(" ");
                }
                Serial.println();
            }
        }
    }

    void printdata()
    {
        if (orig != nullptr)
        {
            Serial.print("Original Data \n");
            Serial.println(orig->name.c_str());
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    Serial.print(orig->data[i][j]);
                    Serial.print(" ");
                }
                Serial.println();
            }
        }
        else
        {
            Serial.print("Data ");
            Serial.println(ptr->name.c_str());
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    Serial.print(ptr->data[i][j]);
                    Serial.print(" ");
                }
                Serial.println();
            }
        }
    }

    void update(float learning_rate)
    {
        if (orig != nullptr) {
            this->ptr = this->orig;
            orig->update(learning_rate);
        } else if (ptr != nullptr) {
            ptr->update(learning_rate);
        }
    }
};