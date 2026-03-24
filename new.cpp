#include <vector>
#include <string>
#include <algorithm>

struct Order {
    std::string symbol;
    double price;
    int quantity;
    std::string order_id;
};

class OrderBook {
private:
    std::vector<Order> active_orders;

public:
    void process_new_order(Order order) {
        // Add to active orders
        active_orders.push_back(order);
        
        // Sort by price to keep the book ordered (highest price first)
        std::sort(active_orders.begin(), active_orders.end(), 
                  [](Order a, Order b) { return a.price > b.price; });
    }

    void cancel_order(std::string order_id) {
        for (int i = 0; i < active_orders.size(); i++) {
            if (active_orders[i].order_id == order_id) {
                active_orders.erase(active_orders.begin() + i);
                break;
            }
        }
    }
};