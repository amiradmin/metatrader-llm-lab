#property strict
#property version   "0.10"
#property description "Lesson 02 - send a rich market snapshot to the Python bridge"

input string BridgeUrl = "http://127.0.0.1:8010/snapshot";
input int    TimerSeconds = 15;

string TimeframeName()
{
   return EnumToString((ENUM_TIMEFRAMES)_Period);
}

string JsonEscape(const string value)
{
   string out = value;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   return out;
}

string IsoTimestamp(const datetime value)
{
   MqlDateTime dt;
   TimeToStruct(value, dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02d",
                       dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
}

void SendSnapshot()
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("HECTOR LLM LAB | SymbolInfoTick failed");
      return;
   }

   MqlRates rates[1];
   if(CopyRates(_Symbol, (ENUM_TIMEFRAMES)_Period, 1, 1, rates) != 1)
   {
      Print("HECTOR LLM LAB | CopyRates failed | error=", GetLastError());
      return;
   }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double spread_points = 0.0;
   if(point > 0.0)
      spread_points = (tick.ask - tick.bid) / point;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);

   string position_status = "NONE";
   string position_type = "NONE";
   double position_volume = 0.0;
   double position_open_price = 0.0;
   double position_profit = 0.0;

   if(PositionSelect(_Symbol))
   {
      position_status = "OPEN";
      long type = PositionGetInteger(POSITION_TYPE);
      position_type = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      position_volume = PositionGetDouble(POSITION_VOLUME);
      position_open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      position_profit = PositionGetDouble(POSITION_PROFIT);
   }

   string body = StringFormat(
      "{\"symbol\":\"%s\",\"timeframe\":\"%s\","
      "\"market\":{\"bid\":%.10f,\"ask\":%.10f,\"spread_points\":%.2f},"
      "\"last_closed_candle\":{\"open\":%.10f,\"high\":%.10f,\"low\":%.10f,\"close\":%.10f,\"tick_volume\":%I64d},"
      "\"account\":{\"balance\":%.8f,\"equity\":%.8f,\"free_margin\":%.8f},"
      "\"position\":{\"status\":\"%s\",\"type\":\"%s\",\"volume\":%.8f,\"open_price\":%.10f,\"profit\":%.8f},"
      "\"timestamp\":\"%s\"}",
      JsonEscape(_Symbol), JsonEscape(TimeframeName()),
      tick.bid, tick.ask, spread_points,
      rates[0].open, rates[0].high, rates[0].low, rates[0].close, rates[0].tick_volume,
      balance, equity, free_margin,
      position_status, position_type, position_volume, position_open_price, position_profit,
      IsoTimestamp(TimeCurrent())
   );

   char request[];
   char response[];
   string response_headers;
   StringToCharArray(body, request, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(request, ArraySize(request)-1);

   string headers = "Content-Type: application/json\r\n";
   ResetLastError();
   int status = WebRequest("POST", BridgeUrl, headers, 5000, request, response, response_headers);

   if(status == -1)
   {
      Print("HECTOR LLM LAB | WebRequest failed | error=", GetLastError());
      return;
   }

   string response_text = CharArrayToString(response, 0, -1, CP_UTF8);
   Print("HECTOR LLM LAB | HTTP=", status,
         " | lesson=02 | symbol=", _Symbol,
         " | timeframe=", TimeframeName(),
         " | response=", response_text);
}

int OnInit()
{
   EventSetTimer((int)MathMax(1, TimerSeconds));
   Print("HECTOR LLM LAB | Lesson 02 started | bridge=", BridgeUrl);
   SendSnapshot();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   SendSnapshot();
}
