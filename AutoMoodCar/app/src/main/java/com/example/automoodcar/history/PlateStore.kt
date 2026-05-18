package com.example.automoodcar.history

object PlateStore {
    val plates = mutableListOf<Pair<String, Long>>()

    fun addPlate(number: String) {
        val now = System.currentTimeMillis()
        plates.removeAll { now - it.second > 15 * 60 * 1000 }
        if (plates.none { it.first == number }) {
            plates.add(number to now)
        }
    }
}