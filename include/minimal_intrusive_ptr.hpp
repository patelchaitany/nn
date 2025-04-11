#ifndef MINIMAL_INTRUSIVE_PTR_HPP
#define MINIMAL_INTRUSIVE_PTR_HPP

#include <cstddef> 

namespace minimal {

class ref_counter {
protected:
    std::size_t ref_count_;

public:
    ref_counter() : ref_count_(0) {}
    
    virtual ~ref_counter() {}
    
    void add_ref() {
        ++ref_count_;
    }
    
    void release() {
        if (--ref_count_ == 0) {
            delete this;
        }
    }
    
    std::size_t use_count() const {
        return ref_count_;
    }
};

template<class T>
class intrusive_ptr {
private:
    T* px;
    
    void add_ref() {
        if (px != nullptr) {
            px->add_ref();
        }
    }
    
    void release() {
        if (px != nullptr) {
            px->release();
            px = nullptr;
        }
    }

public:
    intrusive_ptr() : px(nullptr) {}
    
    intrusive_ptr(T* p) : px(p) {
        add_ref();
    }
    
    intrusive_ptr(const intrusive_ptr& rhs) : px(rhs.px) {
        add_ref();
    }
    
    ~intrusive_ptr() {
        release();
    }
    
    intrusive_ptr& operator=(const intrusive_ptr& rhs) {
        if (this != &rhs) {
            release();
            px = rhs.px;
            add_ref();
        }
        return *this;
    }
    
    intrusive_ptr& operator=(T* rhs) {
        if (px != rhs) {
            release();
            px = rhs;
            add_ref();
        }
        return *this;
    }
    
    T& operator*() const {
        return *px;
    }
    
    T* operator->() const {
        return px;
    }
    
    T* get() const {
        return px;
    }
    
    bool operator!() const {
        return px == nullptr;
    }
    
    operator bool() const {
        return px != nullptr;
    }
    
    friend bool operator!=(const intrusive_ptr<T>& a, std::nullptr_t) {
        return a.px != nullptr;
    }
    
    friend bool operator!=(std::nullptr_t, const intrusive_ptr<T>& a) {
        return a.px != nullptr;
    }
    
    friend bool operator==(const intrusive_ptr<T>& a, std::nullptr_t) {
        return a.px == nullptr;
    }
    
    friend bool operator==(std::nullptr_t, const intrusive_ptr<T>& a) {
        return a.px == nullptr;
    }
};

template<class T>
class intrusive_ref_counter : public ref_counter {
protected:
    intrusive_ref_counter() {}
    intrusive_ref_counter(const intrusive_ref_counter&) {}
    intrusive_ref_counter& operator=(const intrusive_ref_counter&) {
        return *this;
    }
    ~intrusive_ref_counter() {}
};

}

#endif